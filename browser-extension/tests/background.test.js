import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { Script } from "node:vm";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { JSDOM } from "jsdom";
import { retainNativeProgress, proofFailureReport } from "../scripts/live_provider_proof.mjs";
import { setImmediate } from "node:timers/promises";
import { canonicalJson, deriveAccountScope } from "../src/backfill/capture_jobs.js";
import { IndexedDbBackfillStore } from "../src/backfill/storage.js";
import { CaptureStaging } from "../src/capture/staging.js";
import { NativeCaptureNormalizer } from "../src/capture/native.js";
import { memoryOriginStorage, receiverContractPreparation } from "./infra/capture-staging.js";
import { checkpointReceiverState } from "./infra/capture-job-checkpoints.js";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { IDBFactory, IDBKeyRange, IDBObjectStore } from "fake-indexeddb";

const checkpointArtifacts = new Map();
const receiverCheckpointState = checkpointReceiverState();
beforeEach(() => { receiverCheckpointState.clear(); checkpointArtifacts.clear(); });
async function checkpointFixtureResponse(job, options) {
  const result = await receiverCheckpointState.checkpoint(job, options);
  const payload = JSON.parse(result.bytes);
  checkpointArtifacts.set(result.job.checkpoint_digest, { bytes: result.bytes, cutoff: payload.jobs[0]?.cutoff });
  return responseJson({ job: result.job, receipt: result.receipt });
}
function checkpointFixture(payload, sequence = 1) {
  const bytes = canonicalJson(payload);
  const digest = `sha256:${createHash("sha256").update(bytes).digest("hex")}`;
  checkpointArtifacts.set(digest, { bytes, cutoff: payload.jobs[0].cutoff });
  return { sequence, digest, artifact_ref: digest, size_bytes: globalThis.Buffer.byteLength(bytes) };
}
function checkpointArtifactFixtureResponse(url) {
  const path = new globalThis.URL(url).pathname;
  if (!path.includes("/checkpoint-artifacts/")) return null;
  const digest = decodeURIComponent(path.split("/checkpoint-artifacts/")[1]);
  const artifact = checkpointArtifacts.get(digest);
  if (!artifact) return new globalThis.Response(null, { status: 404 });
  return new globalThis.Response(artifact.bytes, { headers: { "Content-Type": "application/json" } });
}
function captureJobRequestBody(options) {
  if (options.headers?.["X-Polylogue-Native"]) return JSON.parse(options.headers["X-Polylogue-Native"]);
  return options.headers?.["X-Polylogue-Checkpoint"]
    ? JSON.parse(options.headers["X-Polylogue-Checkpoint"])
    : options.body ? JSON.parse(options.body) : {};
}

async function deliveryEntries() {
  const entries = [];
  for await (const { entry } of new IndexedDbBackfillStore(globalThis.indexedDB).deliveries()) entries.push(entry);
  return entries;
}

async function makeDeliveriesDue() {
  const owner = new IndexedDbBackfillStore(globalThis.indexedDB);
  for await (const { entry } of owner.deliveries()) {
    await owner.putDelivery({ ...entry, next_attempt_at: new Date(Date.now() - 1000).toISOString() });
  }
}

let messageListener;
let installedListener;
let activatedListener;
let updatedListener;
let removedListener;
let alarmListener;
let stored;
let sessionStored;
let fetchCalls;
let tabs;
//: Loopback origins the mocked browser reports as granted. The manifest
//: grants only the receiver's default port at install; anything else is an
//: optional permission the operator must grant (leak audit L7), so tests
//: that configure another port declare it here.
let grantedOrigins;

let mockGeneration = 0;

function mockPageScript(implementation, { accountHandle = "synthetic-fixture-account" } = {}) {
  return vi.fn(async (details) => {
    if (details.files) return [{ result: undefined }];
    if (details.func && !details.args) return [{ documentId: "synthetic-provider-document", result: true }];
    const result = await implementation(details);
    const request = details.args?.[0];
    if (accountHandle && request?.operation === "identity" && result?.[0]?.result?.ok === true &&
        result[0].result.response?.ok !== false && !result[0].result.response?.accountHandle) {
      result[0].result.response = { accountHandle: `${request.provider}:${accountHandle}` };
    }
    const response = result?.[0]?.result?.response;
    if (response?.ok && typeof response.body === "string") {
      const provider = details.args?.[0]?.provider;
      const owner = { tab_id: details.target.tabId, document_id: "synthetic-provider-document", provider };
      const records = new IndexedDbBackfillStore(globalThis.indexedDB);
      const staging = new CaptureStaging(globalThis.navigator.storage, records);
      const isConversation = ["conversation", "responses", "response-node"].includes(request?.operation);
      let sourceUrl = request?.provider === "claude-ai"
        ? `https://claude.ai/api/organizations/${request.params.organizationId}/chat_conversations/${request.params.nativeId}?tree=True&rendering_mode=messages&render_all_tools=true&consistency=strong`
        : request?.provider === "grok"
          ? `https://grok.com/rest/app-chat/conversations/${request.params.nativeId}${request.operation === "conversation" ? "" : `/${request.operation}`}`
          : `https://chatgpt.com/backend-api/conversation/${request?.params?.nativeId}`;
      if (!isConversation) sourceUrl = `https://synthetic.invalid/${request?.operation || "inventory"}`;
      const ref = await staging.begin(owner, { kind: isConversation ? "native-response" : "provider-inventory",
        source_url: sourceUrl, capture_bundle: request?.capture_bundle, queue_context: request?.queue_context,
        response_metadata: { status: response.status, content_type: response.contentType } });
      if (request?.queue_context) await records.bindNativeAcquisition(ref, owner, request.queue_context);
      const bytes = globalThis.Buffer.from(response.body);
      for (let offset = 0, sequence = 0; offset < bytes.length; offset += 48 * 1024, sequence++) await staging.append(ref, owner, sequence, bytes.subarray(offset, offset + 48 * 1024).toString("base64"));
      await staging.seal(ref, owner);
      const metadata = { ...response };
      delete metadata.body;
      result[0].result.response = { ...metadata, bodyRef: ref };
    }
    return result;
  });
}

function installChromeMock(storagePatch = {}) {
  // Each background instance binds this mock at import. A previous test's
  // instance can still be flushing fire-and-forget persistence (recovery
  // checkpoints, debug logs) when the next test installs a fresh mock; those
  // stale writes must not reach the fresh test's storage.
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { storage: memoryOriginStorage() } });
  const generation = ++mockGeneration;
  const live = () => generation === mockGeneration;
  stored = {
    receiverBaseUrl: "http://127.0.0.1:8875",
    ...storagePatch,
  };
  messageListener = null;
  installedListener = null;
  activatedListener = null;
  updatedListener = null;
  removedListener = null;
  alarmListener = null;
  fetchCalls = [];
  sessionStored = {};
  grantedOrigins = new Set(["http://127.0.0.1:8765/*", "http://127.0.0.1:8875/*"]);
  tabs = [{ id: 42, url: "https://chatgpt.com/?temporary-chat=true", title: "ChatGPT" }];
  globalThis.chrome = {
    // The worker captures this per-instance seam. A stale instance must not
    // forward a request into the next test's fetch stub.
    __polylogueNetwork: async (...args) => {
      if (!live()) throw new Error("stale_background_network");
      const [url, options = {}] = args;
      const path = new globalThis.URL(url).pathname;
      if (path === "/v1/capture-jobs/synthetic-receiver-job" || path === "/v1/capture-jobs/synthetic-receiver-job/adopt") {
        return responseJson({ job: { job_id: "synthetic-receiver-job", provider: "chatgpt", revision: 1, lease_generation: 1 },
          lease: { lease_id: "synthetic-lease", generation: 1, proof: "synthetic-proof" } });
      }
      const response = await globalThis.fetch(...args);
      if (response.ok && options.method === "POST" && (path === "/v1/browser-captures" || path.endsWith("/native/publish"))) {
        // Explicit malformed fields override valid synthetic receipt defaults.
        const body = await response.json();
        const hash = path.endsWith("/native/publish") ? captureJobRequestBody(options).sha256
          : createHash("sha256").update(typeof options.body === "string" ? options.body : new Uint8Array(await options.body.arrayBuffer())).digest("hex");
        response.json = async () => ({ outcome: "accepted", content_hash: hash, submitted_content_hash: hash, ...body });
      }
      return response;
    },
    action: {
      setBadgeBackgroundColor: vi.fn(async () => undefined),
      setBadgeText: vi.fn(async () => undefined),
    },
    permissions: {
      contains: vi.fn(async ({ origins = [] } = {}) => origins.every((origin) => grantedOrigins.has(origin))),
    },
    alarms: {
      create: vi.fn(async () => undefined),
      clear: vi.fn(async () => undefined),
      onAlarm: {
        addListener: vi.fn((fn) => {
          alarmListener = fn;
        }),
      },
    },
    runtime: {
      id: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      getManifest: vi.fn(() => ({ version: "0.1.0" })),
      onInstalled: {
        addListener: vi.fn((fn) => {
          installedListener = fn;
        }),
      },
      onStartup: {
        addListener: vi.fn(),
      },
      onMessage: {
        addListener: vi.fn((fn) => {
          messageListener = fn;
        }),
      },
    },
    scripting: {
      executeScript: mockPageScript(async (details) => {
        if (!details.func) return undefined;
        const request = details.args[0];
        if (request.operation === "identity") {
          return [{ result: { ok: true, response: { accountHandle: `test-account-${request.provider}` } } }];
        }
        const body = request.operation === "inventory"
          ? { items: [{ id: "backfill-1", update_time: 1780000000 }], total: 1 }
          : { id: "backfill-1", mapping: { node: { id: "node", parent: null, message: { id: "message", author: { role: "user" }, content: { content_type: "text", parts: ["synthetic message"] } } } } };
        return [{ result: { ok: true, response: { ok: true, status: 200, contentType: "application/json", body: JSON.stringify(body) } } }];
      }),
    },
    storage: {
      local: {
        get: vi.fn(async (defaults) => (live() ? { ...defaults, ...stored } : { ...defaults })),
        set: vi.fn(async (patch) => {
          if (!live()) return;
          stored = { ...stored, ...patch };
        }),
        remove: vi.fn(async (key) => {
          if (!live()) return;
          const keys = Array.isArray(key) ? key : [key];
          const next = { ...stored };
          for (const item of keys) delete next[item];
          stored = next;
        }),
      },
      session: {
        get: vi.fn(async (defaults) => (live() ? { ...defaults, ...sessionStored } : { ...defaults })),
        set: vi.fn(async (patch) => {
          if (!live()) return;
          sessionStored = { ...sessionStored, ...patch };
        }),
        remove: vi.fn(async (key) => {
          if (!live()) return;
          const keys = Array.isArray(key) ? key : [key];
          for (const item of keys) delete sessionStored[item];
        }),
      },
    },
    tabs: {
      create: vi.fn(async ({ url, active = false }) => {
        const created = { id: 99, url, active, status: "complete" };
        if (live()) tabs = [...tabs, created];
        return created;
      }),
      get: vi.fn(async (tabId) => tabs.find((tab) => tab.id === tabId)),
      update: vi.fn(async (tabId, patch) => {
        const current = tabs.find((tab) => tab.id === tabId);
        const updated = { ...current, ...patch, status: "complete" };
        if (live()) tabs = tabs.map((tab) => (tab.id === tabId ? updated : tab));
        return updated;
      }),
      remove: vi.fn(async (tabId) => {
        if (live()) tabs = tabs.filter((tab) => tab.id !== tabId);
      }),
      onActivated: {
        addListener: vi.fn((fn) => {
          activatedListener = fn;
        }),
      },
      onUpdated: {
        addListener: vi.fn((fn) => {
          updatedListener = fn;
        }),
      },
      onRemoved: {
        addListener: vi.fn((fn) => {
          removedListener = fn;
        }),
      },
      query: vi.fn(async () => tabs),
      sendMessage: vi.fn(async (_tabId, message) => {
        if (message.type === "polylogue.captureIdentity") return { provider_session_id: "temporary:abc" };
        if (message.type === "polylogue.backfill.pageRequest") {
          const body = message.operation === "inventory"
            ? { items: [{ id: "backfill-1", update_time: 1780000000 }], total: 1 }
            : { id: "backfill-1", mapping: { node: { id: "node", parent: null, message: { id: "message", author: { role: "user" }, content: { content_type: "text", parts: ["synthetic message"] } } } } };
          return { ok: true, response: { ok: true, status: 200, contentType: "application/json", body: JSON.stringify(body) } };
        }
        return {
          ok: true,
          captureResult: {
            receiver_request_id: "capture-request-1",
            provider: "chatgpt",
            provider_session_id: "temporary:abc",
          },
          archiveState: {
            receiver_request_id: "state-request-1",
            captured: true,
          },
        };
      }),
    },
  };
  globalThis.fetch = vi.fn(async (url, options = {}) => {
    fetchCalls.push({ url, options });
    const captureJobResponse = captureJobFixtureResponse(url, options);
    if (captureJobResponse) return captureJobResponse;
    if (String(url).endsWith("/v1/browser-captures/capabilities")) {
      return responseJson({ durable_ack_fields: ["receiver_request_id", "content_hash", "submitted_content_hash", "outcome"] }, { requestId: "capability-1" });
    }
    return responseJson({ error: "unexpected_receiver_request" }, { ok: false, status: 500 });
  });
}

function captureJobFixtureResponse(url, options = {}) {
  const artifactResponse = checkpointArtifactFixtureResponse(url);
  if (artifactResponse) return artifactResponse;
  const path = new globalThis.URL(url).pathname;
  if (!path.startsWith("/v1/capture-jobs")) return null;
  const body = captureJobRequestBody(options);
  if (path === "/v1/capture-jobs/capabilities") {
    return responseJson({
      schema: "polylogue.capture-jobs.capabilities.v1", checkpoint_transport: "canonical-artifact-v1",
      protocol_min: 2,
      protocol_max: 2,
      scope_namespace: "cjs1:fixture-stable-namespace",
    });
  }
  if (path === "/v1/capture-jobs/discover") return responseJson({ jobs: [] });
  if (path === "/v1/capture-jobs") {
    const scope = body.scope.kind === "invocation" ? { kind: "invocation", resume_capability: "fixture-resume-capability" } : body.scope;
    return responseJson({ scope, job: receiverCheckpointState.job({
      job_id: "fixture-capture-job", provider: body.provider, scope,
      intent_key: body.intent.intent_key, revision: 0, lease_generation: 0,
    }) }, { status: 201 });
  }
  if (path.endsWith("/adopt")) {
    return responseJson({
      job: receiverCheckpointState.job({
        job_id: "fixture-capture-job", provider: body.provider, scope: body.scope,
        intent_key: "fixture-intent", revision: 1, lease_generation: 1,
      }),
      lease: { lease_id: "fixture-lease", generation: 1, proof: "fixture-proof", expires_at: "2099-01-01T00:00:00Z" },
    });
  }
  if (path.endsWith("/update")) {
    return responseJson({
      job: receiverCheckpointState.job({
        job_id: "fixture-capture-job", provider: body.provider, scope: body.scope,
        intent_key: "fixture-intent", revision: 2, lease_generation: 1,
        lease_expires_at: "2099-01-01T00:00:00Z", checkpoint_sequence: null,
      }),
      receipt: { kind: "capture_job_update", revision: 2 },
    });
  }
  if (path.endsWith("/checkpoint")) {
    return checkpointFixtureResponse({ job_id: "fixture-capture-job", revision: 3 }, options);
  }
  if (path === "/v1/capture-jobs/fixture-capture-job" && (!options.method || options.method === "GET")) {
    return responseJson({ job: receiverCheckpointState.job({ job_id: "fixture-capture-job" }) });
  }
  if (path.endsWith("/native/begin")) {
    receiverCheckpointState.job({ job_id: "fixture-capture-job", native_binding: body.binding });
    return responseJson({ acquisition_id: body.acquisition_id, state: "acquiring" });
  }
  if (path.endsWith("/native/member")) {
    return (async () => {
      expect(options.body.size).toBe(body.size_bytes);
      expect(createHash("sha256").update(globalThis.Buffer.from(await options.body.arrayBuffer())).digest("hex")).toBe(body.sha256);
      return responseJson({ member_name: body.member_name, sha256: body.sha256, size_bytes: body.size_bytes });
    })();
  }
  if (path.endsWith("/native/prepare")) return responseJson({ plan_digest: "sha256:" + "a".repeat(64),
    summary: { title: null, turn_count: 2, attachment_count: 0, session_kind: "standard", needs_follow_up: false } });
  if (path.endsWith("/native/plan")) return responseJson({ plan_digest: "sha256:" + "a".repeat(64), assets: [], after: null });
  if (path.endsWith("/native/finalize")) return responseJson({ sha256: "b".repeat(64), size_bytes: 1234 });
  if (path.endsWith("/native/publish")) {
    const job = receiverCheckpointState.job({ job_id: "fixture-capture-job" });
    return responseJson({ ok: true, provider: job.provider, provider_session_id: job.native_binding.native_id,
      content_hash: body.sha256 });
  }
  return responseJson({ error: "unexpected_capture_job_request" }, { ok: false, status: 500 });
}

async function loadBackground(storagePatch = {}, beforeImport = () => {}) {
  vi.resetModules();
  // Let the previous instance's pending fire-and-forget chains settle before
  // the fresh mock exists; anything later is silenced by the generation guard.
  for (let turn = 0; turn < 25; turn += 1) {
    await new Promise((resolve) => globalThis.setTimeout(resolve, 0));
  }
  globalThis.IDBKeyRange = IDBKeyRange;
  globalThis.indexedDB = new IDBFactory();
  installChromeMock(storagePatch);
  beforeImport();
  await import("../src/background.js");
  expect(messageListener).toBeTypeOf("function");
}

async function sendRuntimeMessage(message, sender = {}) {
  let acknowledge;
  const response = new Promise((resolve) => { acknowledge = resolve; });
  const keepAlive = messageListener(message, sender, acknowledge);
  expect(keepAlive).toBe(true);
  return response;
}

async function captureReceipt(body, options, responseOptions) {
  if (typeof options.body === "string") {
    const descriptor = JSON.parse(options.body);
    if (descriptor.sha256) return responseJson({ ...body, content_hash: descriptor.sha256 }, responseOptions);
  }
  const digest = await globalThis.crypto.subtle.digest("SHA-256", await options.body.arrayBuffer());
  const content_hash = [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
  return responseJson({ ...body, content_hash }, responseOptions);
}

function responseJson(body, { ok = true, status = 200, requestId = "receiver-request-1" } = {}) {
  if (body.job) body = { ...body, job: receiverCheckpointState.job(body.job) };
  for (const job of [...(body.jobs || []), ...(body.job ? [body.job] : [])]) {
    if (!job.checkpoint?.artifact_ref) continue;
    const artifact = checkpointArtifacts.get(job.checkpoint.artifact_ref);
    job.intent ||= { payload: { cutoff: artifact.cutoff } };
    job.checkpoint_updated_at ||= "2026-07-16T10:00:00Z";
  }
  return {
    headers: {
      get: vi.fn((name) => (name === "X-Request-ID" ? requestId : null)),
    },
    json: vi.fn(async () => body),
    ok,
    status,
  };
}

describe("background receiver diagnostics", () => {
  beforeEach(async () => {
    vi.restoreAllMocks();
    vi.useRealTimers();
    await loadBackground();
  });

  it.each(["foreign_submission", "missing_outcome"])("refuses %s in the original foreground receipt boundary", async (fault) => {
    globalThis.fetch = vi.fn(async () => responseJson(
      fault === "foreign_submission" ? { submitted_content_hash: "foreign" } : { outcome: null },
    ));
    const response = await sendRuntimeMessage({ type: "polylogue.capture",
      envelope: { session: { provider: "chatgpt", provider_session_id: "conv-refused", turns: [] } },
    });
    expect(response.ok).toBe(false);
    expect(response.error).toBe("receiver_contract_incompatible");
    expect(stored.polylogueState?.captured).not.toBe(true);
    expect(stored.polylogueSessionLedger["chatgpt:conv-refused"].last_error).toMatch(/^receiver_contract_incompatible:/);
  });

  it("retires a superseded capture without certifying incoming turn counts", async () => {
    stored.polylogueSessionLedger = { "chatgpt:conv-stale": { turn_count: 7, attachment_count: 2 } };
    globalThis.fetch = vi.fn(async () => responseJson({
      ok: true, outcome: "superseded", provider: "chatgpt", provider_session_id: "conv-stale",
      artifact_ref: "chatgpt/conv-stale.json", content_hash: "resident-hash",
      accepted_identities: [],
    }));
    const response = await sendRuntimeMessage({ type: "polylogue.capture",
      envelope: { session: { provider: "chatgpt", provider_session_id: "conv-stale",
        turns: [{ provider_turn_id: "incoming-turn", role: "user", text: "Neutral" }] } },
    });
    expect(response).toMatchObject({ ok: true, outcome: "superseded", captured: false });
    expect(stored.polylogueSessionLedger["chatgpt:conv-stale"]).toMatchObject({
      turn_count: 7, attachment_count: 2, last_error: "receiver_superseded",
    });
    expect(stored.polylogueConversationTimeline["chatgpt:conv-stale"][0]).toMatchObject({
      event: "held_with_reason", detail: "receiver_superseded",
    });
    expect(stored.polylogueState.captured).toBe(false);
    expect(stored.polylogueCaptureQueue?.entries || []).toHaveLength(0);
  });

  it("admits export and exact ACK only from its popup and preserves the paused original delivery", async () => {
    const store = new IndexedDbBackfillStore(globalThis.indexedDB);
    await store.putJob({ id: "export-job", provider: "chatgpt", status: "paused", cooldown_reason: "operator_paused" });
    const staging = new CaptureStaging(globalThis.navigator.storage, store);
    const owner = { tab_id: 42, document_id: "provider-document", provider: "chatgpt" };
    const borrowed = await staging.begin(owner);
    await staging.append(borrowed, owner, 0, globalThis.Buffer.from("original acquired bytes").toString("base64"));
    await staging.seal(borrowed, owner);
    await store.putCapture({ id: "borrowed-raw", kind: "native-acquisition", state: "pending-normalization", source_refs: [borrowed.id] });
    await store.putQueue({ id: "export-item", job_id: "export-job", provider: "chatgpt", native_id: "session", state: "eligible", source_refs: [borrowed.id] });
    const providerSender = { tab: { id: 42, url: "https://chatgpt.com/c/session" }, documentId: "provider-document", url: "https://chatgpt.com/c/session" };
    expect(await sendRuntimeMessage({ type: "polylogue.backfill.export", job_id: "export-job" }, providerSender))
      .toMatchObject({ ok: false, error: "checkpoint_export_sender_invalid" });
    expect(await store.pendingRecoverySnapshot("export-job", "checkpoint-export")).toBeNull();
    const popup = { url: "chrome-extension://aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/src/popup.html" };
    const descriptor = await sendRuntimeMessage({ type: "polylogue.backfill.export", job_id: "export-job" }, popup);
    expect(descriptor.ok).toBe(true);
    expect(descriptor).not.toHaveProperty("ledger"); expect(descriptor).not.toHaveProperty("body");
    expect(await sendRuntimeMessage({ type: "polylogue.backfill.export", job_id: "export-job" }, popup)).toEqual(descriptor);
    const ack = { type: "polylogue.backfill.exportAck", snapshot_id: descriptor.snapshot_id, token: descriptor.token,
      digest: descriptor.digest, size_bytes: descriptor.size_bytes };
    expect(await sendRuntimeMessage(ack, providerSender)).toMatchObject({ ok: false, error: "checkpoint_export_sender_invalid" });
    expect(await sendRuntimeMessage({ ...ack, token: "different-token" }, popup)).toMatchObject({ ok: false, error: "checkpoint_export_owner_mismatch" });
    expect(await sendRuntimeMessage({ ...ack, size_bytes: descriptor.size_bytes + 1 }, popup)).toMatchObject({ ok: false, error: "checkpoint_export_receipt_conflict" });
    expect(await sendRuntimeMessage(ack, popup)).toMatchObject({ ok: true, outcome: "exported" });
    expect(await sendRuntimeMessage(ack, popup)).toMatchObject({ ok: true, outcome: "exported" });
    expect(await store.getQueue("export-item")).toMatchObject({ source_refs: [borrowed.id], state: "eligible" });
    expect(await store.getCapture("borrowed-raw")).toBeDefined();
    expect(await store.getJob("export-job")).toMatchObject({ status: "paused", cooldown_reason: "operator_paused" });
  });

  it("refuses switched-account acquisition before provider traffic without changing existing custody", async () => {
    tabs[0].url = "https://chatgpt.com/c/session";
    stored.polylogueReceiverPairing = { state: "online", receiver_id: "account-race-receiver",
      api_schema: "polylogue-browser-capture/v1", endpoint: "http://127.0.0.1:8875" };
    const namespace = "cjs1:account-race-namespace";
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      fetchCalls.push({ url, options });
      if (String(url).endsWith("/v1/status")) return responseJson({ ok: true, receiver_id: "account-race-receiver", api_schema: "polylogue-browser-capture/v1" });
      if (String(url).endsWith("/v1/capture-jobs/capabilities")) return responseJson({ schema: "polylogue.capture-jobs.capabilities.v1",
        checkpoint_transport: "canonical-artifact-v1", protocol_min: 2, protocol_max: 2, scope_namespace: namespace });
      throw new Error("unexpected_provider_or_receiver_request");
    });
    const store = new IndexedDbBackfillStore(globalThis.indexedDB);
    const staging = new CaptureStaging(globalThis.navigator.storage, store);
    const body = await staging.prepare({ session: { turns: [{ text: "unacknowledged original custody" }] } });
    const scope = await deriveAccountScope(namespace, "chatgpt", "account-A");
    await store.putJob({ id: "job", provider: "chatgpt", account_scope: scope, status: "running",
      execution_owner: "worker", execution_generation: 1, execution_expires_at_ms: Date.now() + 60_000 });
    const item = { id: "item", job_id: "job", provider: "chatgpt", native_id: "session", state: "leased",
      lease_owner: "worker", lease_expires_at_ms: Date.now() + 60_000, body_ref: body.ref };
    await store.putQueue(item);
    const result = await sendRuntimeMessage({ type: "polylogue.asset.begin", provider: "chatgpt", request_id: "account-race",
      kind: "native-response", source_url: "https://chatgpt.com/backend-api/conversation/session", account_handle: "account-B",
      queue_context: { itemId: "item", jobId: "job", owner: "worker", generation: 1, nativeId: "session" } },
      { tab: tabs[0], documentId: "owned-document" });
    expect(result).toMatchObject({ ok: false, error: "capture_job_account_scope_mismatch" });
    expect(await store.getQueue("item")).toMatchObject(item);
    expect((await store.getJob("job")).account_scope).toBe(scope);
    expect(await (await staging.file(body.ref)).text()).toContain("unacknowledged original custody");
    expect(fetchCalls.filter((call) => String(call.url).startsWith("https://"))).toEqual([]);
    expect(JSON.stringify(stored)).not.toContain("account-A");
    expect(JSON.stringify(stored)).not.toContain("account-B");
  });

  it("preserves an ACKed cache pin on a tab lookup fault and releases it only on positive document loss", async () => {
    stored.polylogueAmbientSettings = { enabled: true, automatic_capture_enabled: false, disabled_sites: [] };
    const owner = { tab_id: 42, document_id: "cache-document", provider: "chatgpt" };
    const store = new IndexedDbBackfillStore(globalThis.indexedDB);
    const staging = new CaptureStaging(globalThis.navigator.storage, store);
    const raw = await staging.begin(owner, { kind: "native-response" });
    const payload = JSON.stringify({ id: "cached-session", mapping: {} });
    await staging.append(raw, owner, 0, globalThis.Buffer.from(payload).toString("base64")); await staging.seal(raw, owner);
    await store.pinNativeCache({ owner, provider: "chatgpt", nativeId: "cached-session", rawRef: raw,
      headers: { id: "cached-session" }, observedAt: "2026-01-01T00:00:00Z", acquisitionSequence: 1 });
    await store.retireNativeAcquisition(raw.id);
    await staging.save({ ...await staging.metadata(raw.id), retired: true });
    globalThis.chrome.tabs.get.mockRejectedValue(new Error("synthetic_tab_lookup_unavailable"));
    vi.resetModules(); await import("../src/background.js");
    await vi.waitFor(() => expect(stored.polylogueDebugLog).toEqual(expect.arrayContaining([
      expect.objectContaining({ stage: "capture_staging_recovery_failed", error: "synthetic_tab_lookup_unavailable" }),
    ])));
    expect(await store.captureReferences(raw.id)).toBe(true);
    expect(await (await staging.file(raw.id)).text()).toBe(payload);
    globalThis.chrome.tabs.get.mockRejectedValue(new Error("No tab with id: 42."));
    vi.resetModules(); await import("../src/background.js");
    await vi.waitFor(async () => expect(await store.captureReferences(raw.id)).toBe(false));
    await vi.waitFor(async () => expect(staging.file(raw.id)).rejects.toMatchObject({ code: "capture_staging_missing_bytes" }));
    expect(fetchCalls.filter((call) => String(call.url).startsWith("https://"))).toEqual([]);
  });

  it("returns the exact acquired revision while keeping a newer current-page cache authoritative", async () => {
    tabs[0].url = "https://chatgpt.com/c/session";
    const owner = { tab_id: 42, document_id: "owned-document", provider: "chatgpt" };
    const store = new IndexedDbBackfillStore(globalThis.indexedDB);
    const staging = new CaptureStaging(globalThis.navigator.storage, store);
    const revisions = [];
    for (const text of ["older acquired evidence", "newer acquired evidence"]) {
      const ref = await staging.begin(owner, { kind: "native-response", source_url: "https://chatgpt.com/backend-api/conversation/session" });
      await staging.append(ref, owner, 0, globalThis.Buffer.from(JSON.stringify({ conversation_id: "session", update_time: 1, mapping: {}, title: text })).toString("base64"));
      await staging.seal(ref, owner);
      revisions.push(ref);
    }
    const newerMeta = await staging.metadata(revisions[1].id);
    await store.pinNativeCache({ owner, provider: "chatgpt", nativeId: "session", rawRef: revisions[1], headers: { conversation_id: "session", update_time: 1 },
      observedAt: newerMeta.created_at, acquisitionSequence: newerMeta.acquisition_sequence });
    const result = await sendRuntimeMessage({ type: "polylogue.nativeCaptureHeader", provider: "chatgpt", raw_ref: revisions[0] },
      { tab: tabs[0], documentId: owner.document_id });
    expect(result.ok).toBe(true);
    expect(result.capture.bodyRef).toEqual(revisions[0]);
    expect(result.headers).not.toHaveProperty("title");
    expect(result.cache.capture.bodyRef).toEqual(revisions[1]);
    expect(await store.getCapture(`raw:${revisions[0].id}`)).toMatchObject({ state: "pending-normalization" });
    expect(await (await staging.file(revisions[0].id)).text()).toContain("older acquired evidence");
    expect(fetchCalls.filter((call) => String(call.url).startsWith("https://"))).toEqual([]);
  });

  it.each(["capabilities", "prepare"].flatMap(boundary => ["success", "cancel"].map(outcome => [boundary, outcome])))("distinguishes the original preparation await despite identical markers: %s/%s", async (boundary, outcome) => {
    stored.polylogueReceiverPairing = { state: "online", receiver_id: "rx-1", api_schema: "polylogue-browser-capture/v1", endpoint: stored.receiverBaseUrl };
    tabs[0].url = "https://chatgpt.com/c/session";
    const owner = { tab_id: 42, document_id: "owned-document", provider: "chatgpt" };
    const store = new IndexedDbBackfillStore(globalThis.indexedDB);
    const staging = new CaptureStaging(globalThis.navigator.storage, store);
    const ref = await staging.begin(owner, { kind: "native-response", source_url: "https://chatgpt.com/backend-api/conversation/session" });
    const raw = JSON.stringify({ id: "session", mapping: {} });
    await staging.append(ref, owner, 0, globalThis.Buffer.from(raw).toString("base64"));
    await staging.seal(ref, owner);
    const operations = []; let release; let pending = 0; let memberBytes;
    const originalFetch = globalThis.fetch;
    globalThis.fetch = vi.fn(async (url, options) => {
      const path = new globalThis.URL(url).pathname;
      const operation = path === "/v1/capture-jobs/capabilities" ? "capabilities" : path.split("/native/")[1];
      if (operation) operations.push(operation);
      if (operation === "member") memberBytes = await options.body.text();
      if (operation === boundary) {
        pending++;
        try { await new Promise(resolve => { release = resolve; }); }
        finally { pending--; }
      }
      if (path === "/v1/status") return responseJson({ ok: true, receiver_id: "rx-1", api_schema: "polylogue-browser-capture/v1" });
      return captureJobFixtureResponse(url, options) || originalFetch(url, options);
    });
    const normalization = sendRuntimeMessage({ type: "polylogue.normalizeNativeCapture", provider: "chatgpt", raw_ref: ref,
      native_id: "session", native_request_id: "polylogue-native-fetch-1-original" }, { tab: tabs[0], documentId: owner.document_id });
    let result; normalization.then(value => { result = value; });
    await vi.waitFor(() => expect(release).toBeDefined());
    await vi.waitFor(() => expect(stored.polylogueDebugLog).toEqual(expect.arrayContaining([
      expect.objectContaining({ stage: "native_preparation_progress", phase: "native_prepare", state: "BEGIN", native_request_id: "polylogue-native-fetch-1-original" }),
    ])));
    expect(result).toBeUndefined(); expect(pending).toBe(1);
    expect(stored.polylogueDebugLog.some(row => row.phase === "native_prepare" && row.state === "END")).toBe(false);
    if (boundary === "capabilities") {
      expect(operations).toEqual(["capabilities"]); expect(memberBytes).toBeUndefined();
    } else {
      expect(operations).toEqual(["capabilities", "begin", "member", "prepare"]);
      expect(memberBytes).toBe(raw);
    }
    let cancellation; let cancellationSettled = false;
    if (outcome === "cancel") {
      cancellation = sendRuntimeMessage({ type: "polylogue.cancelNativeCapture", provider: "chatgpt", raw_ref: ref }, { tab: tabs[0], documentId: owner.document_id });
      cancellation.then(() => { cancellationSettled = true; });
      await setImmediate();
      expect(cancellationSettled).toBe(false); expect(result).toBeUndefined(); expect(pending).toBe(1);
    }
    release();
    result = await normalization;
    if (cancellation) expect(await cancellation).toMatchObject({ ok: true, outcome: "cancelled" });
    expect(pending).toBe(0);
    expect(result.ok).toBe(outcome === "success");
    if (outcome === "cancel") expect(result.outcome).toBe("cancelled");
    else expect(result.envelope.receiver_native).toHaveProperty("sha256");
  });

  it.each([
    ["admission", "success"],
    ...["prepare", "plan", "finalize"].flatMap(boundary => ["success", "failure", "cancel"].map(outcome => [boundary, outcome])),
    ["prepare", "diagnostic_failure"], ["prepare", "missing_id"], ["prepare", "malformed_id"],
  ])("records private fixed progress before the actual suspended native %s request: %s", async (boundary, outcome) => {
    stored.polylogueReceiverPairing = { state: "online", receiver_id: "rx-1", api_schema: "polylogue-browser-capture/v1", endpoint: stored.receiverBaseUrl };
    tabs[0].url = "https://chatgpt.com/c/session";
    const owner = { tab_id: 42, document_id: "owned-document", provider: "chatgpt" };
    const store = new IndexedDbBackfillStore(globalThis.indexedDB);
    const staging = new CaptureStaging(globalThis.navigator.storage, store);
    const ref = await staging.begin(owner, { kind: "native-response", source_url: "https://chatgpt.com/backend-api/conversation/session" });
    await staging.append(ref, owner, 0, globalThis.Buffer.from(JSON.stringify({ id: "session", mapping: {} })).toString("base64"));
    await staging.seal(ref, owner);
    if (outcome === "diagnostic_failure") {
      const originalSet = globalThis.chrome.storage.local.set;
      globalThis.chrome.storage.local.set = async patch => {
        if (patch.polylogueDebugLog?.[0]?.stage === "native_preparation_progress") throw new Error("synthetic debug write refusal");
        return originalSet(patch);
      };
    }
    let release; const originalFetch = globalThis.fetch;
    globalThis.fetch = vi.fn(async (url, options) => {
      const path = new globalThis.URL(url).pathname;
      if (path.endsWith(`/native/${boundary}`) || (boundary === "admission" && path === "/v1/status")) await new Promise((resolve, reject) => { release = () => outcome === "failure" ? reject(new Error("synthetic native request refusal")) : resolve(); });
      if (path === "/v1/status") return responseJson({ ok: true, receiver_id: "rx-1", api_schema: "polylogue-browser-capture/v1" });
      return captureJobFixtureResponse(url, options) || originalFetch(url, options);
    });
    const normalization = sendRuntimeMessage({ type: "polylogue.normalizeNativeCapture", provider: "chatgpt", raw_ref: ref, native_id: "session", native_request_id: outcome === "missing_id" ? undefined : outcome === "malformed_id" ? "staging-uuid" : "polylogue-native-fetch-1-original" }, { tab: tabs[0], documentId: owner.document_id });
    let earlyResponse; normalization.then(value => { earlyResponse = value; });
    await vi.waitFor(() => expect(release, JSON.stringify(earlyResponse)).toBeDefined());
    const phase = { admission: "normalize_admission", prepare: "native_prepare", plan: "native_assets", finalize: "native_finalize" }[boundary];
    if (!["diagnostic_failure", "missing_id", "malformed_id"].includes(outcome)) await vi.waitFor(() => expect(stored.polylogueDebugLog).toEqual(expect.arrayContaining([
      expect.objectContaining({stage:"native_preparation_progress",phase,state:"BEGIN",acquisition_ref:ref.id}),
    ])));
    const begun = stored.polylogueDebugLog.find(row => row.phase === phase && row.state === "BEGIN");
    if (!["diagnostic_failure", "missing_id", "malformed_id"].includes(outcome)) expect(Object.keys(begun).sort()).toEqual(["acquisition_ref","at","native_request_id","phase","stage","state"]);
    else expect(begun).toBeUndefined();
    expect(stored.polylogueDebugLog.some(row => row.phase === phase && row.state === "END")).toBe(false);
    const cancellation = outcome === "cancel" ? sendRuntimeMessage({ type: "polylogue.cancelNativeCapture", provider: "chatgpt", raw_ref: ref }, { tab: tabs[0], documentId: owner.document_id }) : null;
    if (cancellation) await setImmediate();
    release();
    const result = await normalization;
    if (cancellation) await cancellation;
    expect(result.ok).toBe(["success", "diagnostic_failure", "missing_id", "malformed_id"].includes(outcome));
    if (outcome === "success") await vi.waitFor(() => expect(stored.polylogueDebugLog).toEqual(expect.arrayContaining([
      expect.objectContaining({stage:"native_preparation_progress",phase:"native_finalize",state:"END",acquisition_ref:ref.id}),
    ])));
    if (["failure", "cancel"].includes(outcome)) expect(stored.polylogueDebugLog.some(row => row.phase === phase && row.state === "END")).toBe(false);
  });

  it.each(["same_clock", "interleaved", "missing_id", "malformed_id"])("carries ORIGINAL MAIN request through replaced headers, private log and strict report: %s", async schedule => {
    stored.polylogueAmbientSettings = { enabled: true, automatic_capture_enabled: false, disabled_sites: [] };
    stored.polylogueReceiverPairing = { state: "online", receiver_id: "rx-1", api_schema: "polylogue-browser-capture/v1", endpoint: stored.receiverBaseUrl };
    tabs[0].url = "https://chatgpt.com/c/session";
    const owner = { tab_id: 42, document_id: "owned-document", provider: "chatgpt" };
    const store = new IndexedDbBackfillStore(globalThis.indexedDB);
    const staging = new CaptureStaging(globalThis.navigator.storage, store);
    const ref = await staging.begin(owner, { kind: "native-response", source_url: "https://chatgpt.com/backend-api/conversation/session" });
    await staging.append(ref, owner, 0, globalThis.Buffer.from(JSON.stringify({ id: "session", mapping: {} })).toString("base64"));
    await staging.seal(ref, owner);
    const fixedClock = Date.now(); vi.spyOn(Date, "now").mockReturnValue(fixedClock);
    const allStorageListeners = new Set(); const pages = []; const normalizations = []; const releases = [];
    const originalFetch = globalThis.fetch;
    globalThis.fetch = vi.fn(async (url, options) => {
      if (new globalThis.URL(url).pathname.endsWith("/native/prepare")) await new Promise(resolve => { releases.push(resolve); });
      if (new globalThis.URL(url).pathname === "/v1/status") return responseJson({ ok: true, receiver_id: "rx-1", api_schema: "polylogue-browser-capture/v1" });
      return captureJobFixtureResponse(url, options) || originalFetch(url, options);
    });
    const originalSet = globalThis.chrome.storage.local.set;
    globalThis.chrome.storage.local.set = async patch => {
      const changes = Object.fromEntries(Object.entries(patch).map(([key, value]) => [key, { oldValue: stored[key], newValue: value }]));
      await originalSet(patch);
      for (const listener of allStorageListeners) listener(changes, "local");
    };
    try {
      for (let index = 0; index < (schedule === "interleaved" ? 2 : 1); index++) {
        const dom = new JSDOM("<!doctype html><title>Neutral capture fixture</title>", { url: tabs[0].url, runScripts: "outside-only" });
        const listeners = []; const storageListeners = new Set(); const page = { dom, storageListeners, registrationListeners: null, requestId: null, capture: null };
        pages.push(page); dom.window.Date.now = () => fixedClock;
        dom.window.chrome = {
          storage: { onChanged: { addListener: listener => { storageListeners.add(listener); allStorageListeners.add(listener); }, removeListener: listener => { storageListeners.delete(listener); allStorageListeners.delete(listener); } } },
          runtime: { id: globalThis.chrome.runtime.id, getManifest: globalThis.chrome.runtime.getManifest,
            onMessage: { addListener: listener => listeners.push(listener) },
            sendMessage: async message => {
              const request = { ...message };
              if (message.type === "polylogue.normalizeNativeCapture") {
                expect(storageListeners.size).toBe(2); // Registration plus original preparation observer, before runtime send.
                expect(message.native_request_id).toBe(page.requestId);
                expect(message.raw_ref).toEqual(ref); // Header response actually replaced the acquisition descriptor.
                normalizations.push(message);
                if (schedule === "missing_id") delete request.native_request_id;
                if (schedule === "malformed_id") request.native_request_id = "staging-uuid";
              }
              const result = await sendRuntimeMessage(request, { tab: tabs[0], documentId: owner.document_id });
              // The real header handler is retained; the neutral fixture presents
              // an identical owned raw revision for both concurrent responses.
              return message.type === "polylogue.nativeCaptureHeader" && result?.ok
                ? { ...result, capture: { ...result.capture, bodyRef: ref } } : result;
            } },
        };
        dom.window.fetch = async input => new globalThis.Response(JSON.stringify(new globalThis.URL(String(input)).pathname === "/api/auth/session" ? { detail: "not_found" } : { id: "session", mapping: {} }), { status: new globalThis.URL(String(input)).pathname === "/api/auth/session" ? 404 : 200, headers: { "content-type": "application/json" } });
        dom.window.postMessage = data => {
          if (data.type === `polylogue.page.v2.${dom.window.chrome.runtime.id}.chatgpt.nativeFetchRequest`) page.requestId = data.requestId;
          dom.window.queueMicrotask(() => dom.window.dispatchEvent(new dom.window.MessageEvent("message", { source: dom.window, origin: dom.window.location.origin, data })));
        };
        const context = dom.getInternalVMContext();
        for (const file of ["content/asset_stream.js", "content/chatgpt_bridge.js", "common.js", "content/chatgpt.js"]) new Script(readFileSync(resolve(dirname(fileURLToPath(import.meta.url)), `../src/${file}`), "utf8")).runInContext(context);
        page.registrationListeners = new Set(storageListeners);
        page.dispatch = message => new Promise((resolve, reject) => {
          if (!listeners.some(listener => listener(message, {}, resolve) === true)) reject(new Error("synthetic content listener missing"));
        });
        page.capture = page.dispatch({ type: "polylogue.capturePage", deferReceiver: true });
      }
      await vi.waitFor(() => expect(releases.length).toBe(pages.length));
      expect(normalizations).toHaveLength(pages.length);
      expect(new Set(pages.map(page => page.requestId)).size).toBe(pages.length);
      for (const page of pages) expect(page.requestId).toMatch(/^polylogue-native-fetch-\d+-[a-z0-9]+$/);
      if (!["missing_id", "malformed_id"].includes(schedule)) for (const page of pages) {
        await vi.waitFor(() => expect(stored.polylogueDebugLog).toEqual(expect.arrayContaining([expect.objectContaining({ stage: "native_preparation_progress", phase: "normalize_admission", state: "BEGIN", native_request_id: page.requestId, acquisition_ref: ref.id })])));
      }
      const cancellations = pages.map(page => page.dispatch({ type: "polylogue.cancelCapture" }));
      expect(allStorageListeners.size).toBe(pages.length);
      for (const page of pages) expect(page.storageListeners).toEqual(page.registrationListeners);
      for (const release of releases) release();
      await Promise.all(cancellations);
      for (const page of pages) {
        const progress = await page.capture;
        expect(progress).toMatchObject({ ok: false, outcome: "cancelled" });
        const rows = progress.native_progress.filter(row => row.source === "background_debug_log");
        if (["missing_id", "malformed_id"].includes(schedule)) expect(rows).toEqual([]);
        else expect(rows).toContainEqual({ stage: "normalize_admission", state: "BEGIN", source: "background_debug_log" });
        retainNativeProgress(progress.native_progress);
        expect(proofFailureReport("capture", new Error("synthetic refusal")).native_progress).toEqual(progress.native_progress);
        expect(JSON.stringify(progress.native_progress)).not.toContain(page.requestId);
        expect(JSON.stringify(progress.native_progress)).not.toContain(ref.id);
        const snapshot = JSON.stringify(progress.native_progress);
        // A real late private write after original cancellation cannot revive an observer.
        await globalThis.chrome.storage.local.set({ polylogueDebugLog: [{ at: new Date().toISOString(), stage: "native_preparation_progress", phase: "native_finalize", state: "END", native_request_id: page.requestId, acquisition_ref: ref.id }] });
        expect(JSON.stringify(progress.native_progress)).toBe(snapshot);
      }
    } finally {
      const cancellations = pages.map(page => page.dispatch?.({ type: "polylogue.cancelCapture" }));
      for (const release of releases) release();
      await Promise.allSettled(cancellations);
      await Promise.allSettled(pages.map(page => page.capture));
      for (const page of pages) { page.dom.window.dispatchEvent(new page.dom.window.Event("pagehide")); page.dom.window.close(); }
      expect(allStorageListeners.size).toBe(0);
      vi.restoreAllMocks();
    }
  });

  it("uses the supported Claude native id header without inventing a uuid", async () => {
    tabs[0].url = "https://claude.ai/chat/native-id";
    const owner = { tab_id: 42, document_id: "claude-id-document", provider: "claude-ai" };
    const store = new IndexedDbBackfillStore(globalThis.indexedDB);
    const staging = new CaptureStaging(globalThis.navigator.storage, store);
    const ref = await staging.begin(owner, { kind: "native-response", source_url: "https://claude.ai/api/organizations/fixture/chat_conversations/native-id" });
    const raw = JSON.stringify({ id: "native-id", chat_messages: [] });
    await staging.append(ref, owner, 0, globalThis.Buffer.from(raw).toString("base64"));
    await staging.seal(ref, owner);
    const result = await sendRuntimeMessage({ type: "polylogue.nativeCaptureHeader", provider: "claude-ai", raw_ref: ref },
      { tab: tabs[0], documentId: owner.document_id });
    expect(result).toMatchObject({ ok: true, capture: { nativeId: "native-id", bodyRef: ref }, headers: { id: "native-id" } });
    expect(result.headers.uuid).toBeUndefined();
    expect(await (await staging.file(ref.id)).text()).toBe(raw);
    expect(fetchCalls.filter((call) => String(call.url).startsWith("https://"))).toEqual([]);
  });

  it("restores identity-validated sealed native evidence without a page replay or provider read", async () => {
    tabs[0].url = "https://chatgpt.com/c/retained-session";
    const owner = { tab_id: 42, document_id: "retained-document", provider: "chatgpt" };
    const store = new IndexedDbBackfillStore(globalThis.indexedDB);
    const staging = new CaptureStaging(globalThis.navigator.storage, store);
    const refs = [];
    for (const id of ["retained-session", "unrelated-session"]) {
      const ref = await staging.begin(owner, { kind: "native-response", source_url: `https://chatgpt.com/backend-api/conversation/${id}` });
      await staging.append(ref, owner, 0, globalThis.Buffer.from(JSON.stringify({ id, mapping: {} })).toString("base64"));
      await staging.seal(ref, owner); refs.push(ref);
    }
    globalThis.fetch = vi.fn(async (url) => {
      if (String(url).startsWith("https://chatgpt.com/")) throw new Error("unexpected-provider-read");
      return responseJson({});
    });
    const sender = { tab: tabs[0], documentId: owner.document_id };
    const restored = await sendRuntimeMessage({ type: "polylogue.restoreNativeCapture", provider: "chatgpt", native_id: "retained-session", request_id: globalThis.crypto.randomUUID() }, sender);
    expect(restored).toMatchObject({ ok: true, capture: { bodyRef: refs[0], headers: { id: "retained-session" } } });
    expect(await store.captureReferences(refs[0].id)).toBe(true);
    // Unselected acquired evidence remains pending; restoration never deletes it.
    expect(await store.captureReferences(refs[1].id)).toBe(true);
    expect(await store.getCapture(`raw:${refs[1].id}`)).toMatchObject({ state: "pending-normalization" });
    expect(globalThis.fetch.mock.calls.filter(([url]) => String(url).startsWith("https://chatgpt.com/"))).toHaveLength(0);
    const writes = globalThis.navigator.storage.writes.length;
    const repeated = await sendRuntimeMessage({ type: "polylogue.restoreNativeCapture", provider: "chatgpt", native_id: "retained-session", request_id: globalThis.crypto.randomUUID() }, sender);
    expect(repeated.capture.bodyRef).toEqual(refs[0]);
    expect(globalThis.navigator.storage.writes.length).toBe(writes);
    const lostDocument = await sendRuntimeMessage({ type: "polylogue.restoreNativeCapture", provider: "chatgpt", native_id: "retained-session", request_id: globalThis.crypto.randomUUID() }, { ...sender, documentId: "different-document" });
    expect(lostDocument).toEqual({ ok: true, capture: null });
  });

  it("cancels and drains native recovery's active file reader with exact document ownership", async () => {
    tabs[0].url = "https://chatgpt.com/c/cancel-session";
    const owner = { tab_id: 42, document_id: "recovery-document", provider: "chatgpt" };
    const store = new IndexedDbBackfillStore(globalThis.indexedDB);
    const staging = new CaptureStaging(globalThis.navigator.storage, store);
    const ref = await staging.begin(owner, { kind: "native-response" });
    await staging.append(ref, owner, 0, globalThis.Buffer.from(JSON.stringify({ id: "cancel-session", mapping: {} })).toString("base64"));
    await staging.seal(ref, owner);
    let reading; const started = new Promise((resolve) => { reading = resolve; }); let cancelled = false;
    const directory = globalThis.navigator.storage.directory;
    const getHandle = directory.getFileHandle.bind(directory);
    directory.getFileHandle = async (name, options) => {
      const handle = await getHandle(name, options);
      if (name !== `${ref.id}.bytes`) return handle;
      const file = await handle.getFile();
      return { ...handle, getFile: async () => ({ size: file.size, stream: () => new globalThis.ReadableStream({
        pull() { reading(); return new Promise(() => {}); },
        cancel() { cancelled = true; },
      }) }) };
    };
    const sender = { tab: tabs[0], documentId: owner.document_id };
    const requestId = "cancel-recovery-request";
    const recovery = sendRuntimeMessage({ type: "polylogue.restoreNativeCapture", provider: "chatgpt", native_id: "cancel-session", request_id: requestId }, sender);
    await started;
    expect(await sendRuntimeMessage({ type: "polylogue.restoreNativeCapture", provider: "chatgpt", native_id: "cancel-session", request_id: requestId }, sender))
      .toMatchObject({ ok: false, error: "capture_staging_request_conflict" });
    const wrongOwner = await sendRuntimeMessage({ type: "polylogue.cancelNativeRecovery", provider: "chatgpt", request_id: requestId }, { ...sender, documentId: "unrelated-document" });
    expect(wrongOwner.ok).toBe(false);
    expect(cancelled).toBe(false);
    const cancellation = await sendRuntimeMessage({ type: "polylogue.cancelNativeRecovery", provider: "chatgpt", request_id: requestId }, sender);
    expect(cancellation).toMatchObject({ ok: true, outcome: "cancelled" });
    expect(cancelled).toBe(true);
    expect(await recovery).toMatchObject({ ok: false, error: "capture_cancelled" });
    expect(await store.captureReferences(ref.id)).toBe(true);
    expect(await store.getCapture(`raw:${ref.id}`)).toMatchObject({ state: "pending-normalization" });
    directory.getFileHandle = getHandle;
    expect(await (await staging.file(ref.id)).text()).toBe(JSON.stringify({ id: "cancel-session", mapping: {} }));
  });

  it("cancels every concurrent native reader of the exact raw revision before acknowledging", async () => {
    tabs[0].url = "https://chatgpt.com/c/concurrent-readers";
    const owner = { tab_id: 42, document_id: "owned-document", provider: "chatgpt" };
    const store = new IndexedDbBackfillStore(globalThis.indexedDB);
    const staging = new CaptureStaging(globalThis.navigator.storage, store);
    const ref = await staging.begin(owner, { kind: "native-response" });
    await staging.append(ref, owner, 0, globalThis.Buffer.from('{"id":"concurrent-readers","mapping":{}}').toString("base64"));
    await staging.seal(ref, owner);
    let readers = 0; let cancelled = 0; let ready;
    const started = new Promise((resolve) => { ready = resolve; });
    const directory = globalThis.navigator.storage.directory;
    const original = directory.getFileHandle.bind(directory);
    directory.getFileHandle = async (name, options) => {
      const handle = await original(name, options);
      if (name !== `${ref.id}.bytes`) return handle;
      const file = await handle.getFile();
      return { ...handle, getFile: async () => ({ size: file.size, stream: () => new globalThis.ReadableStream({
        pull() { if (++readers === 2) ready(); return new Promise(() => {}); },
        cancel() { cancelled += 1; },
      }) }) };
    };
    const sender = { tab: tabs[0], documentId: owner.document_id };
    const first = sendRuntimeMessage({ type: "polylogue.nativeCaptureHeader", provider: "chatgpt", raw_ref: ref }, sender);
    const second = sendRuntimeMessage({ type: "polylogue.nativeCaptureHeader", provider: "chatgpt", raw_ref: ref }, sender);
    await started;
    expect(await sendRuntimeMessage({ type: "polylogue.cancelNativeCapture", provider: "chatgpt", raw_ref: ref }, sender))
      .toMatchObject({ ok: true, outcome: "cancelled" });
    expect(cancelled).toBe(2);
    expect(await first).toMatchObject({ ok: false, error: "capture_cancelled" });
    expect(await second).toMatchObject({ ok: false, error: "capture_cancelled" });
    expect((await staging.metadata(ref.id)).state).toBe("sealed");
  });

  it("sends a request id to the receiver and stores the echoed id", async () => {
    globalThis.fetch = vi.fn(async (url, options) => {
      fetchCalls.push({ url, options });
      return captureReceipt({
        ok: true,
        provider: "chatgpt",
        provider_session_id: "conv-123",
        artifact_ref: "chatgpt/conv-123.json",
      }, options);
    });

    const response = await sendRuntimeMessage({
      type: "polylogue.capture",
      envelope: { polylogue_capture_kind: "browser_llm_session" },
    });

    expect(response.receiver_request_id).toBe("receiver-request-1");
    expect(fetchCalls).toHaveLength(1);
    expect(fetchCalls[0].url).toBe("http://127.0.0.1:8875/v1/browser-captures");
    expect(fetchCalls[0].options.headers.Authorization).toBeUndefined();
    expect(fetchCalls[0].options.headers["Content-Type"]).toBe("application/json");
    expect(fetchCalls[0].options.headers["X-Request-ID"]).toMatch(/^polylogue-ext-/);
    expect(fetchCalls[0].options.headers["X-Polylogue-Extension-Contract"])
      .toBe("canonical-capture-mission-control-v1");
    expect(stored.polylogueState.last_receiver_request_id).toBe("receiver-request-1");
    expect(stored.polylogueState.last_capture.receiver_request_id).toBe("receiver-request-1");
  });

  it("keeps a reused native delivery exclusively owned until its upload cancellation drains", async () => {
    stored.polylogueReceiverPairing = { state: "online", receiver_id: "rx-delivery",
      api_schema: "polylogue-browser-capture/v1", endpoint: stored.receiverBaseUrl };
    const store = new IndexedDbBackfillStore(globalThis.indexedDB);
    const staging = new CaptureStaging(globalThis.navigator.storage, store);
    const owner = { tab_id: 42, document_id: "capture-owner", provider: "chatgpt" };
    const ref = await staging.begin(owner, { kind: "native-response", source_url: "https://chatgpt.com/backend-api/conversation/session" });
    await staging.append(ref, owner, 0, globalThis.Buffer.from(JSON.stringify({ conversation_id: "session", mapping: {
      node: { message: { id: "message", author: { role: "assistant" }, content: { parts: ["retained evidence"] } } },
    } })).toString("base64"));
    await staging.seal(ref, owner);
    const envelope = await new NativeCaptureNormalizer({ staging, store, prepareNative: receiverContractPreparation(staging, store) }).normalize({ provider: "chatgpt", rawRef: ref,
      nativeId: "session", extensionVersion: "0.3.0", instanceId: "synthetic-preparation-instance", signal: new globalThis.AbortController().signal });
    const sender = { tab: tabs[0], documentId: owner.document_id };
    globalThis.fetch = vi.fn(async (url) => {
      if (String(url).endsWith("/native/publish")) throw new TypeError("Failed to fetch");
      return responseJson({ ok: true, receiver_id: stored.polylogueReceiverPairing.receiver_id,
        api_schema: stored.polylogueReceiverPairing.api_schema });
    });
    expect(await sendRuntimeMessage({ type: "polylogue.capture", envelope }, sender)).toMatchObject({ queued: true });
    let uploads = 0; let drained = false;
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      if (!String(url).endsWith("/native/publish")) return responseJson({ ok: true,
        receiver_id: stored.polylogueReceiverPairing.receiver_id, api_schema: stored.polylogueReceiverPairing.api_schema });
      uploads += 1;
      try { await new Promise((resolve, reject) => {
        options.signal.addEventListener("abort", () => reject(options.signal.reason), { once: true });
        if (options.signal.aborted) reject(options.signal.reason);
      }); } finally { drained = true; }
    });
    const capture = sendRuntimeMessage({ type: "polylogue.capture", request_id: "reused-delivery", envelope }, sender);
    await vi.waitFor(() => expect(uploads).toBe(1));
    await makeDeliveriesDue();
    expect(await sendRuntimeMessage({ type: "polylogue.retryCaptureQueue" })).toMatchObject({ drained: 0, remaining: 1 });
    expect(uploads).toBe(1);
    expect(await sendRuntimeMessage({ type: "polylogue.cancelCaptureDelivery", request_id: "reused-delivery" }, sender)).toMatchObject({ ok: true });
    expect(drained).toBe(true);
    expect(await capture).toMatchObject({ error: "capture_cancelled" });
    expect(await deliveryEntries()).toHaveLength(1);
  });

  it("shares one immutable native upload and lets one caller cancel without retiring the other caller's evidence", async () => {
    stored.polylogueReceiverPairing = { state: "online", receiver_id: "rx-delivery",
      api_schema: "polylogue-browser-capture/v1", endpoint: stored.receiverBaseUrl };
    const store = new IndexedDbBackfillStore(globalThis.indexedDB);
    const staging = new CaptureStaging(globalThis.navigator.storage, store);
    const owner = { tab_id: 42, document_id: "capture-owner", provider: "chatgpt" };
    const ref = await staging.begin(owner, { kind: "native-response", source_url: "https://chatgpt.com/backend-api/conversation/session" });
    await staging.append(ref, owner, 0, globalThis.Buffer.from(JSON.stringify({ conversation_id: "session", mapping: {
      node: { message: { id: "message", author: { role: "assistant" }, content: { parts: ["shared evidence"] } } },
    } })).toString("base64"));
    await staging.seal(ref, owner);
    const envelope = await new NativeCaptureNormalizer({ staging, store, prepareNative: receiverContractPreparation(staging, store) }).normalize({ provider: "chatgpt", rawRef: ref,
      nativeId: "session", extensionVersion: "0.3.0", instanceId: "synthetic-preparation-instance", signal: new globalThis.AbortController().signal });
    const sender = { tab: tabs[0], documentId: owner.document_id };
    let finishUpload; let uploads = 0; let uploadSignal; let uploadBody;
    globalThis.fetch = vi.fn(async (url, options) => {
      if (!String(url).endsWith("/native/publish")) return responseJson({ ok: true,
        receiver_id: stored.polylogueReceiverPairing.receiver_id, api_schema: stored.polylogueReceiverPairing.api_schema });
      uploads += 1; uploadSignal = options.signal; uploadBody = options.body;
      await new Promise((resolve) => { finishUpload = resolve; });
      return captureReceipt({ ok: true, provider: "chatgpt", provider_session_id: "session" }, options);
    });
    const first = sendRuntimeMessage({ type: "polylogue.capture", request_id: "shared-first", envelope }, sender);
    await vi.waitFor(() => expect(uploads).toBe(1));
    const { IndexedDbBackfillStore: RuntimeCaptureStore } = await import("../src/backfill/storage.js");
    const deliveryRoot = vi.spyOn(RuntimeCaptureStore.prototype, "foregroundDeliveryRoot");
    const second = sendRuntimeMessage({ type: "polylogue.capture", request_id: "shared-second", envelope }, sender);
    await vi.waitFor(() => expect(deliveryRoot).toHaveBeenCalledWith(envelope.capture_record_ref));
    await deliveryRoot.mock.results.at(-1).value;
    // Finish the durable-root promise continuations before changing physical ownership.
    await setImmediate();
    expect(await sendRuntimeMessage({ type: "polylogue.cancelCaptureDelivery", request_id: "shared-second" }, sender)).toMatchObject({ ok: true });
    expect(await second).toMatchObject({ ok: false, error: "capture_cancelled" });
    expect(uploadSignal.aborted).toBe(false);
    expect(JSON.parse(uploadBody).sha256).toBe(envelope.receiver_native.sha256);
    expect(await (await staging.file(ref.id)).text()).toContain("shared evidence");
    expect(await deliveryEntries()).toHaveLength(1);
    finishUpload();
    expect(await first).toMatchObject({ ok: true });
    expect(uploads).toBe(1);
    await vi.waitFor(async () => expect(await deliveryEntries()).toHaveLength(0));
  });

  it("keeps late native delivery admission outside a cancelled physical upload until it drains", async () => {
    stored.polylogueReceiverPairing = { state: "online", receiver_id: "rx-delivery",
      api_schema: "polylogue-browser-capture/v1", endpoint: stored.receiverBaseUrl };
    const store = new IndexedDbBackfillStore(globalThis.indexedDB);
    const staging = new CaptureStaging(globalThis.navigator.storage, store);
    const owner = { tab_id: 42, document_id: "capture-owner", provider: "chatgpt" };
    const ref = await staging.begin(owner, { kind: "native-response", source_url: "https://chatgpt.com/backend-api/conversation/session" });
    await staging.append(ref, owner, 0, globalThis.Buffer.from(JSON.stringify({ conversation_id: "session", mapping: {
      node: { message: { id: "message", author: { role: "assistant" }, content: { parts: ["closing evidence"] } } },
    } })).toString("base64"));
    await staging.seal(ref, owner);
    const envelope = await new NativeCaptureNormalizer({ staging, store, prepareNative: receiverContractPreparation(staging, store) }).normalize({ provider: "chatgpt", rawRef: ref,
      nativeId: "session", extensionVersion: "0.3.0", instanceId: "synthetic-preparation-instance", signal: new globalThis.AbortController().signal });
    const sender = { tab: tabs[0], documentId: owner.document_id };
    let finishDrain; let closing = false; let uploads = 0;
    globalThis.fetch = vi.fn(async (url, options) => {
      if (!String(url).endsWith("/native/publish")) return responseJson({ ok: true,
        receiver_id: stored.polylogueReceiverPairing.receiver_id, api_schema: stored.polylogueReceiverPairing.api_schema });
      uploads += 1;
      if (uploads === 1) {
        await new Promise((resolve) => options.signal.addEventListener("abort", () => { closing = true; resolve(); }, { once: true }));
        await new Promise((resolve) => { finishDrain = resolve; });
        throw options.signal.reason;
      }
      return captureReceipt({ ok: true, provider: "chatgpt", provider_session_id: "session" }, options);
    });
    const first = sendRuntimeMessage({ type: "polylogue.capture", request_id: "closing-first", envelope }, sender);
    await vi.waitFor(() => expect(uploads).toBe(1));
    const cancellation = sendRuntimeMessage({ type: "polylogue.cancelCaptureDelivery", request_id: "closing-first" }, sender);
    await vi.waitFor(() => expect(closing).toBe(true));
    const { IndexedDbBackfillStore: RuntimeCaptureStore } = await import("../src/backfill/storage.js");
    const deliveryRoot = vi.spyOn(RuntimeCaptureStore.prototype, "foregroundDeliveryRoot");
    const late = sendRuntimeMessage({ type: "polylogue.capture", request_id: "closing-late", envelope }, sender);
    await vi.waitFor(() => expect(deliveryRoot).toHaveBeenCalledWith(envelope.capture_record_ref));
    await deliveryRoot.mock.results.at(-1).value;
    // Finish the durable-root promise continuations before changing physical ownership.
    await setImmediate();
    expect(uploads).toBe(1);
    expect(await deliveryEntries()).toHaveLength(1);
    finishDrain();
    expect(await cancellation).toMatchObject({ ok: true });
    expect(await first).toMatchObject({ ok: false, error: "capture_cancelled" });
    expect(await late).toMatchObject({ ok: true });
    expect(uploads).toBe(2);
    await vi.waitFor(async () => expect(await deliveryEntries()).toHaveLength(0));
  });

  it("returns its durable ACK while retirement waits behind another delivery's physical upload", async () => {
    stored.polylogueReceiverPairing = { state: "online", receiver_id: "rx-delivery",
      api_schema: "polylogue-browser-capture/v1", endpoint: stored.receiverBaseUrl };
    const sender = { tab: tabs[0], documentId: "capture-owner" };
    globalThis.fetch = vi.fn(async (url) => {
      if (String(url).endsWith("/v1/browser-captures")) throw new TypeError("Failed to fetch");
      return responseJson({ ok: true, receiver_id: stored.polylogueReceiverPairing.receiver_id,
        api_schema: stored.polylogueReceiverPairing.api_schema });
    });
    expect(await sendRuntimeMessage({ type: "polylogue.capture", envelope: { session: {
      provider: "chatgpt", provider_session_id: "other-delivery", turns: [{ text: "other evidence" }],
    } } }, sender)).toMatchObject({ queued: true });
    let finishOwn; let finishOther; let ownStarted = false; let otherStarted = false;
    const uploadedIds = [];
    globalThis.fetch = vi.fn(async (url, options) => {
      if (!String(url).endsWith("/v1/browser-captures")) return responseJson({ ok: true,
        receiver_id: stored.polylogueReceiverPairing.receiver_id, api_schema: stored.polylogueReceiverPairing.api_schema });
      const id = JSON.parse(await options.body.text()).session.provider_session_id;
      uploadedIds.push(id);
      await new Promise((resolve) => {
        if (id === "own-delivery") { ownStarted = true; finishOwn = resolve; }
        else { otherStarted = true; finishOther = resolve; }
      });
      return captureReceipt({ ok: true, provider: "chatgpt", provider_session_id: id }, options);
    });
    const own = sendRuntimeMessage({ type: "polylogue.capture", request_id: "ack-before-retirement", envelope: { session: {
      provider: "chatgpt", provider_session_id: "own-delivery", turns: [{ text: "own evidence" }],
    } } }, sender);
    await vi.waitFor(() => expect(ownStarted).toBe(true));
    await makeDeliveriesDue();
    const retry = sendRuntimeMessage({ type: "polylogue.retryCaptureQueue" });
    await vi.waitFor(() => expect(otherStarted).toBe(true));
    finishOwn();
    expect(await own).toMatchObject({ ok: true });
    const rows = await deliveryEntries();
    expect(rows.some((entry) => entry.summary.providerSessionId === "own-delivery")).toBe(true);
    const ownEntry = rows.find((entry) => entry.summary.providerSessionId === "own-delivery");
    expect(await new CaptureStaging(globalThis.navigator.storage, new IndexedDbBackfillStore(globalThis.indexedDB)).metadata(ownEntry.body_ref))
      .toMatchObject({ state: "receiver-acknowledged" });
    finishOther();
    expect(await retry).toMatchObject({ remaining: 0 });
    expect(uploadedIds).toEqual(["own-delivery", "other-delivery"]);
    await vi.waitFor(async () => expect(await deliveryEntries()).toHaveLength(0));
  });

  it("cancels capture preparation queued behind an unrelated progressing retry without publishing it", async () => {
    stored.polylogueReceiverPairing = { state: "online", receiver_id: "rx-delivery",
      api_schema: "polylogue-browser-capture/v1", endpoint: stored.receiverBaseUrl };
    const sender = { tab: tabs[0], documentId: "capture-owner" };
    globalThis.fetch = vi.fn(async (url) => {
      if (String(url).endsWith("/v1/browser-captures")) throw new TypeError("Failed to fetch");
      return responseJson({ ok: true, receiver_id: stored.polylogueReceiverPairing.receiver_id,
        api_schema: stored.polylogueReceiverPairing.api_schema });
    });
    expect(await sendRuntimeMessage({ type: "polylogue.capture", envelope: { session: {
      provider: "chatgpt", provider_session_id: "unrelated-retry", turns: [{ text: "original evidence" }],
    } } }, sender)).toMatchObject({ queued: true });
    await makeDeliveriesDue();
    let finishUpload; let uploads = 0;
    globalThis.fetch = vi.fn(async (url, options) => {
      if (!String(url).endsWith("/v1/browser-captures")) return responseJson({ ok: true,
        receiver_id: stored.polylogueReceiverPairing.receiver_id, api_schema: stored.polylogueReceiverPairing.api_schema });
      uploads += 1;
      await new Promise((resolve) => { finishUpload = resolve; });
      return captureReceipt({ ok: true, provider: "chatgpt", provider_session_id: "unrelated-retry" }, options);
    });
    const retry = sendRuntimeMessage({ type: "polylogue.retryCaptureQueue" });
    await vi.waitFor(() => expect(uploads).toBe(1));
    const capture = sendRuntimeMessage({ type: "polylogue.capture", request_id: "waiting-preparation", envelope: { session: {
      provider: "chatgpt", provider_session_id: "cancelled-before-preparation", turns: [{ text: "must not publish" }],
    } } }, sender);
    expect(await sendRuntimeMessage({ type: "polylogue.cancelCaptureDelivery", request_id: "waiting-preparation" }, sender))
      .toMatchObject({ ok: true });
    expect(await capture).toMatchObject({ ok: false, error: "capture_cancelled" });
    expect(uploads).toBe(1);
    finishUpload();
    expect(await retry).toMatchObject({ drained: 1, remaining: 0 });
    expect(await deliveryEntries()).toHaveLength(0);
    expect(uploads).toBe(1);
  });

  it("cancels and drains an owned receiver upload while retaining unacknowledged bytes", async () => {
    stored.polylogueReceiverPairing = { state: "online", receiver_id: "rx-delivery",
      api_schema: "polylogue-browser-capture/v1", endpoint: stored.receiverBaseUrl };
    let started = false; let drained = false;
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      if (!String(url).endsWith("/v1/browser-captures")) return responseJson({
        ok: true, receiver_id: stored.polylogueReceiverPairing.receiver_id, api_schema: stored.polylogueReceiverPairing.api_schema,
      });
      started = true;
      expect(options.body.size).toBeGreaterThan(0);
      try {
        await new Promise((resolve, reject) => {
          options.signal.addEventListener("abort", () => reject(options.signal.reason), { once: true });
          if (options.signal.aborted) reject(options.signal.reason);
        });
      } finally { drained = true; }
    });
    const sender = { tab: tabs[0], documentId: "capture-owner" };
    const capture = sendRuntimeMessage({ type: "polylogue.capture", request_id: "owned-delivery",
      envelope: { session: { provider: "chatgpt", provider_session_id: "cancelled-session", turns: [{ text: "retained evidence" }] } },
    }, sender);
    await vi.waitFor(() => expect(started).toBe(true));
    expect(await sendRuntimeMessage({ type: "polylogue.cancelCaptureDelivery", request_id: "owned-delivery" },
      { ...sender, documentId: "other-document" })).toMatchObject({ ok: false, error: "capture_delivery_owner_mismatch" });
    expect(drained).toBe(false);
    expect(await sendRuntimeMessage({ type: "polylogue.cancelCaptureDelivery", request_id: "owned-delivery" }, sender)).toMatchObject({ ok: true });
    expect(drained).toBe(true);
    expect(await capture).toMatchObject({ ok: false, error: "capture_cancelled" });
    const entries = await deliveryEntries();
    expect(entries).toHaveLength(1); expect(entries[0]).toMatchObject({ held: true, next_attempt_at: null });
    expect(await (await new CaptureStaging(globalThis.navigator.storage).file(entries[0].body_ref)).text()).toContain("retained evidence");
  });

  it("coalesces concurrent capture attribution into one stable service-worker instance", async () => {
    globalThis.fetch = vi.fn(async (url, options) => {
      fetchCalls.push({ url, options });
      return captureReceipt({ ok: true, provider: "chatgpt", provider_session_id: "conv-123" }, options);
    });

    await Promise.all([
      sendRuntimeMessage({
        type: "polylogue.capture",
        envelope: {
          provenance: { extension_instance_id: "untrusted-content-script" },
          session: { provider: "chatgpt", provider_session_id: "conv-123" },
        },
      }),
      sendRuntimeMessage({
        type: "polylogue.capture",
        envelope: { session: { provider: "chatgpt", provider_session_id: "conv-124" } },
      }),
    ]);

    const first = JSON.parse(await fetchCalls[0].options.body.text());
    const second = JSON.parse(await fetchCalls[1].options.body.text());
    expect(first.provenance.extension_instance_id).not.toBe("untrusted-content-script");
    expect(first.provenance.extension_instance_id).toBe(second.provenance.extension_instance_id);
    expect(stored.polylogueExtensionInstanceId).toBe(first.provenance.extension_instance_id);
  });

  it("records a direct manual capture as pending timeline evidence", async () => {
    globalThis.fetch = vi.fn(async (_url, options) => captureReceipt({
      provider: "chatgpt",
      provider_session_id: "conv-manual",
      state: "spooled_only",
      artifact_ref: "chatgpt/conv-manual.json",
    }, options));

    await sendRuntimeMessage({
      type: "polylogue.capture",
      reason: "content_script_capture",
      envelope: {
        session: {
          provider: "chatgpt",
          provider_session_id: "conv-manual",
          turns: [{ role: "user" }],
        },
      },
    });

    expect(stored.polylogueState.archive_state).toEqual({ state: "spooled_only" });
    expect(stored.polylogueConversationTimeline["chatgpt:conv-manual"][0]).toMatchObject({
      event: "captured",
      reason: "content_script_capture",
      detail: "spooled_only",
    });
    expect(stored.polylogueSessionLedger["chatgpt:conv-manual"].archive_state).toEqual({ state: "spooled_only" });
  });

  it("records an inactive direct capture as catching up in its ledger", async () => {
    tabs = [
      { id: 1, url: "https://chatgpt.com/c/conv-active", active: true },
      { id: 2, url: "https://chatgpt.com/c/conv-inactive", active: false },
    ];
    stored.polylogueReceiverPairing = {
      state: "online", receiver_id: "rx-inactive", api_schema: "polylogue-browser-capture/v1",
    };
    globalThis.fetch = vi.fn(async (url, options) => String(url).endsWith("/v1/status") ? responseJson({
      ok: true, receiver_id: "rx-inactive", api_schema: "polylogue-browser-capture/v1",
    }) : captureReceipt({
      provider: "chatgpt",
      provider_session_id: "conv-inactive",
      state: "spooled_only",
    }, options));

    await sendRuntimeMessage({
      type: "polylogue.capture",
      envelope: { session: { provider: "chatgpt", provider_session_id: "conv-inactive", turns: [] } },
    }, { tab: tabs[1] });

    expect(stored.polylogueSessionLedger["chatgpt:conv-inactive"].archive_state).toEqual({ state: "spooled_only" });
    expect(stored.polylogueState).toBeUndefined();
  });

  it("records a non-retryable capture rejection as a held decision", async () => {
    globalThis.fetch = vi.fn(async () => responseJson({ error: "invalid capture" }, { ok: false, status: 400 }));

    const response = await sendRuntimeMessage({
      type: "polylogue.capture",
      reason: "auto_capture_missing",
      envelope: { session: { provider: "chatgpt", provider_session_id: "conv-rejected", turns: [] } },
    });

    expect(response).toMatchObject({ ok: false, error: "invalid capture" });
    expect(stored.polylogueConversationTimeline["chatgpt:conv-rejected"][0]).toMatchObject({
      event: "held_with_reason",
      reason: "auto_capture_missing",
      detail: "capture_rejected",
    });
    expect(stored.polylogueSessionLedger["chatgpt:conv-rejected"].last_error).toBe("invalid capture");
  });

  it("does not let an inactive rejection replace the active conversation card", async () => {
    tabs = [
      { id: 1, url: "https://chatgpt.com/c/conv-active", active: true },
      { id: 2, url: "https://chatgpt.com/c/conv-rejected", active: false },
    ];
    stored.polylogueState = { online: true, provider: "chatgpt", provider_session_id: "conv-active", archive_state: { state: "archived" } };
    stored.polylogueReceiverPairing = {
      state: "online", receiver_id: "rx-inactive", api_schema: "polylogue-browser-capture/v1",
    };
    globalThis.fetch = vi.fn(async (url) => responseJson(
      String(url).endsWith("/v1/status")
        ? { ok: true, receiver_id: "rx-inactive", api_schema: "polylogue-browser-capture/v1" }
        : { error: "invalid capture" },
      String(url).endsWith("/v1/status") ? {} : { ok: false, status: 400 },
    ));

    await sendRuntimeMessage({
      type: "polylogue.capture",
      envelope: { session: { provider: "chatgpt", provider_session_id: "conv-rejected", turns: [] } },
    }, { tab: tabs[1] });

    expect(stored.polylogueState.provider_session_id).toBe("conv-active");
    expect(stored.polylogueConversationTimeline["chatgpt:conv-rejected"][0].detail).toBe("capture_rejected");
  });

  it("keeps receiver request id on error state", async () => {
    globalThis.fetch = vi.fn(async () =>
      responseJson({ error: "unauthorized" }, { ok: false, status: 401, requestId: "reject-42" }),
    );

    const response = await sendRuntimeMessage({ type: "polylogue.status" });

    expect(response).toMatchObject({
      ok: false,
      error: "unauthorized",
      receiver_request_id: "reject-42",
    });
    expect(response.receiver_pairing).toBeNull();
    expect(stored.polylogueState.online).toBe(false);
    expect(stored.polylogueState.last_receiver_request_id).toBe("reject-42");
  });

  it("refuses an unpaired content capture before a receiver POST", async () => {
    delete stored.polylogueReceiverPairing;

    const response = await sendRuntimeMessage({
      type: "polylogue.capture",
      envelope: { session: { provider: "chatgpt", provider_session_id: "unpaired-content", turns: [] } },
    }, { tab: tabs[0] });

    expect(response).toMatchObject({ ok: false, error: "receiver_unpaired" });
    expect(fetchCalls).toHaveLength(0);
    expect(stored.polylogueCaptureQueue?.entries || []).toHaveLength(0);
    expect(stored.polylogueSessionLedger["chatgpt:unpaired-content"].last_error).toBe("receiver_unpaired");
  });

  it("records a held decision when archive-state cannot be checked", async () => {
    tabs = [{ id: 42, url: "https://chatgpt.com/c/conv-offline", title: "ChatGPT" }];
    globalThis.fetch = vi.fn(async () => {
      throw new TypeError("Failed to fetch");
    });

    activatedListener({ tabId: 42 });

    await vi.waitFor(() => expect(stored.polylogueConversationTimeline["chatgpt:conv-offline"]?.[0]).toMatchObject({
      event: "held_with_reason",
      reason: "tab_activated",
      detail: "archive_state_check_failed",
    }));
    expect(stored.polylogueState.active_page_state).toBe("receiver_error");
  });

  it("refreshes the active conversation instead of discarding its archive identity", async () => {
    tabs = [{ id: 42, url: "https://chatgpt.com/c/conv-status", title: "ChatGPT" }];
    globalThis.fetch = vi.fn(async (url) => {
      if (String(url).endsWith("/v1/status")) {
        return responseJson({
          ok: true,
          api_schema: "polylogue-browser-capture/v1",
          receiver_id: "rx-status",
        }, { requestId: "status-1" });
      }
      return responseJson({
        provider: "chatgpt",
        provider_session_id: "conv-status",
        state: "spooled_only",
        captured: false,
      });
    });

    const response = await sendRuntimeMessage({ type: "polylogue.status", reason: "popup_open" });

    expect(response.state).toBe("spooled_only");
    expect(stored.polylogueState).toMatchObject({
      provider: "chatgpt",
      provider_session_id: "conv-status",
      archive_state: { state: "spooled_only" },
    });
    expect(stored.polylogueSessionLedger["chatgpt:conv-status"].archive_state.state).toBe("spooled_only");
  });

  it("propagates content-script archive state into the multi-tab ledger", async () => {
    globalThis.fetch = vi.fn(async () => responseJson({
      provider: "chatgpt",
      provider_session_id: "conv-content",
      state: "stale",
      captured: false,
    }));

    await sendRuntimeMessage({
      type: "polylogue.archiveState",
      provider: "chatgpt",
      provider_session_id: "conv-content",
    });

    expect(stored.polylogueSessionLedger["chatgpt:conv-content"].archive_state.state).toBe("stale");
    expect(globalThis.chrome.action.setBadgeText.mock.calls.at(-1)[0]).toEqual({ text: "…" });
  });

  it("preserves capture metadata when content refreshes its archive state", async () => {
    globalThis.fetch = vi.fn(async (url, options) => String(url).endsWith("/v1/browser-captures")
      ? captureReceipt({ provider: "chatgpt", provider_session_id: "conv-metadata" }, options)
      : responseJson({ provider: "chatgpt", provider_session_id: "conv-metadata", state: "spooled_only", captured: false }));

    await sendRuntimeMessage({
      type: "polylogue.capture",
      envelope: {
        session: {
          provider: "chatgpt",
          provider_session_id: "conv-metadata",
          provider_meta: { capture_fidelity: "dom_degraded" },
          turns: [{ role: "user" }, { role: "assistant" }],
        },
      },
    });
    await sendRuntimeMessage({
      type: "polylogue.archiveState",
      provider: "chatgpt",
      provider_session_id: "conv-metadata",
    });

    expect(stored.polylogueState).toMatchObject({
      capture_mode: "dom_degraded",
      turn_count: 2,
      archive_state: { state: "spooled_only" },
    });
  });

  it("does not let an inactive tab update replace the active conversation card state", async () => {
    tabs = [
      { id: 1, url: "https://chatgpt.com/c/conv-active", title: "Active", active: true },
      { id: 2, url: "https://chatgpt.com/c/conv-inactive", title: "Inactive", active: false },
    ];
    stored.polylogueState = {
      online: true,
      provider: "chatgpt",
      provider_session_id: "conv-active",
      archive_state: { state: "archived" },
    };
    globalThis.fetch = vi.fn(async () => responseJson({
      provider: "chatgpt",
      provider_session_id: "conv-inactive",
      state: "archived",
      captured: true,
    }));

    updatedListener(2, { status: "complete" }, tabs[1]);

    await vi.waitFor(() => expect(stored.polylogueSessionLedger["chatgpt:conv-inactive"]?.archive_state?.state).toBe("archived"));
    expect(stored.polylogueState.provider_session_id).toBe("conv-active");
  });

  it("does not let a delayed prior conversation overwrite same-tab navigation", async () => {
    tabs = [{ id: 1, url: "https://chatgpt.com/c/conv-a", title: "A", active: true }];
    let resolveA;
    globalThis.fetch = vi.fn(async (url) => {
      if (String(url).includes("conv-a")) return new Promise((resolve) => { resolveA = resolve; });
      return responseJson({ provider: "chatgpt", provider_session_id: "conv-b", state: "archived", captured: true });
    });

    updatedListener(1, { status: "complete" }, { id: 1, url: "https://chatgpt.com/c/conv-a", title: "A" });
    tabs[0] = { id: 1, url: "https://chatgpt.com/c/conv-b", title: "B", active: true };
    updatedListener(1, { url: "https://chatgpt.com/c/conv-b" }, tabs[0]);

    await vi.waitFor(() => expect(stored.polylogueState?.provider_session_id).toBe("conv-b"));
    resolveA(responseJson({ provider: "chatgpt", provider_session_id: "conv-a", state: "archived", captured: true }));
    await vi.waitFor(() => expect(stored.polylogueSessionLedger?.["chatgpt:conv-a"]?.archive_state?.state).toBe("archived"));
    expect(stored.polylogueState.provider_session_id).toBe("conv-b");
  });

  it("does not restore a delayed conversation after same-tab navigation to a new page", async () => {
    tabs = [{ id: 1, url: "https://chatgpt.com/c/conv-a", title: "A", active: true }];
    let resolveA;
    let fetchCount = 0;
    globalThis.fetch = vi.fn(async () => {
      fetchCount += 1;
      if (fetchCount === 1) return new Promise((resolve) => { resolveA = resolve; });
      return new Promise(() => {});
    });

    updatedListener(1, { status: "complete" }, tabs[0]);
    tabs[0] = { id: 1, url: "https://chatgpt.com/new", title: "New", active: true };
    updatedListener(1, { url: "https://chatgpt.com/new" }, tabs[0]);

    await vi.waitFor(() => expect(globalThis.fetch).toHaveBeenCalledTimes(2));
    resolveA(responseJson({ provider: "chatgpt", provider_session_id: "conv-a", state: "archived", captured: true }));
    await new Promise((resolve) => globalThis.setTimeout(resolve, 20));
    expect(stored.polylogueState?.provider_session_id).not.toBe("conv-a");
  });

  it("holds a missing conversation when the tab navigates before auto-capture", async () => {
    tabs = [{ id: 1, url: "https://chatgpt.com/c/conv-a", title: "A", active: true }];
    let resolveArchiveState;
    globalThis.fetch = vi.fn(async () => new Promise((resolve) => { resolveArchiveState = resolve; }));

    updatedListener(1, { status: "complete" }, tabs[0]);
    await vi.waitFor(() => expect(globalThis.fetch).toHaveBeenCalledTimes(1));
    tabs[0] = { id: 1, url: "https://chatgpt.com/c/conv-b", title: "B", active: true };
    resolveArchiveState(responseJson({ provider: "chatgpt", provider_session_id: "conv-a", state: "missing", captured: false }));

    await vi.waitFor(() => expect(stored.polylogueConversationTimeline["chatgpt:conv-a"]?.[0]).toMatchObject({
      event: "held_with_reason",
      detail: "tab_navigation_changed",
    }));
    expect(globalThis.chrome.tabs.sendMessage).not.toHaveBeenCalled();
  });

  it("uses Grok query conversation identity for archive polling and the ledger", async () => {
    tabs = [{ id: 77, url: "https://grok.com/?conversation=query-77", title: "Grok", active: true }];
    globalThis.fetch = vi.fn(async (url) => {
      fetchCalls.push({ url });
      return responseJson({ provider: "grok", provider_session_id: "query-77", state: "archived", captured: true });
    });

    activatedListener({ tabId: 77 });

    await vi.waitFor(() => expect(stored.polylogueSessionLedger["grok:query-77"]?.archive_state?.state).toBe("archived"));
    expect(fetchCalls[0].url).toContain("provider=grok&provider_session_id=query-77");
  });

  it("does not invent a conversation identity for the X home timeline", async () => {
    tabs = [{ id: 78, url: "https://x.com/home", title: "Home", active: true }];
    globalThis.fetch = vi.fn(async (url) => {
      fetchCalls.push({ url });
      return responseJson({ ok: true });
    });

    activatedListener({ tabId: 78 });

    await vi.waitFor(() => expect(fetchCalls).toHaveLength(1));
    expect(fetchCalls[0].url).toBe("http://127.0.0.1:8875/v1/status");
    expect(stored.polylogueSessionLedger).toBeUndefined();
    expect(globalThis.chrome.tabs.sendMessage).not.toHaveBeenCalled();
  });

  it("records a local popup capture failure as a held decision without offline state", async () => {
    tabs = [{ id: 42, url: "https://chatgpt.com/c/conv-local-failure", title: "ChatGPT", active: true }];

    await sendRuntimeMessage({
      type: "polylogue.capturePageFailed",
      tab_id: 42,
      tab_url: tabs[0].url,
      error: "no_turns",
    });

    expect(stored.polylogueConversationTimeline["chatgpt:conv-local-failure"][0]).toMatchObject({
      event: "held_with_reason",
      detail: "content_capture_failed",
    });
    expect(stored.polylogueState).toMatchObject({ online: true, error: "no_turns" });
  });

  it("refuses unpaired freshness inventory before any provider operation", async () => {
    stored.polylogueReceiverPairing = null;
    let release;
    const get = globalThis.chrome.storage.local.get.getMockImplementation();
    globalThis.chrome.storage.local.get.mockImplementation((defaults) => {
      if (Object.hasOwn(defaults, "polylogueReceiverPairing")) return new Promise(resolve => { release = () => resolve({ polylogueReceiverPairing: null }); });
      return get(defaults);
    });
    alarmListener({ name: "polylogueCaptureFreshnessSweep" });
    await vi.waitFor(() => expect(release).toBeTypeOf("function"));
    expect(globalThis.chrome.scripting.executeScript).not.toHaveBeenCalled();
    release();
    await new Promise(resolve => globalThis.setTimeout(resolve, 0));
    expect(globalThis.chrome.scripting.executeScript).not.toHaveBeenCalled();
    expect(fetchCalls.filter(call => String(call.url).startsWith("https://"))).toEqual([]);
  });

  it.each(["archived", "spooled_only"])("installs freshness observers in existing %s tabs without recapturing", async (state) => {
    expect(installedListener).toBeTypeOf("function");
    tabs = [{ id: 42, url: "https://chatgpt.com/c/conv-installed", title: "ChatGPT" }];
    globalThis.fetch = vi.fn(async () => responseJson({
      provider: "chatgpt",
      provider_session_id: "conv-installed",
      state,
      captured: true,
    }));

    installedListener();

    await vi.waitFor(() => expect(stored.polylogueSessionLedger["chatgpt:conv-installed"]?.archive_state)
      .toMatchObject({ state }));
    expect(globalThis.chrome.tabs.sendMessage).not.toHaveBeenCalled();
    expect(globalThis.chrome.scripting.executeScript).toHaveBeenCalledWith({
      target: { tabId: 42 }, files: ["src/content/asset_stream.js", "src/content/chatgpt_bridge.js"], world: "MAIN",
    });
    expect(globalThis.chrome.scripting.executeScript).toHaveBeenCalledWith({
      target: { tabId: 42 }, files: expect.arrayContaining(["src/common.js", "src/content/chatgpt.js"]),
    });
  });

  it("routes backfill inventory through an existing provider page without creating a tab", async () => {
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      fetchCalls.push({ url, options });
      const captureJobResponse = captureJobFixtureResponse(url, options);
      if (captureJobResponse) return captureJobResponse;
      if (String(url).endsWith("/v1/browser-captures/capabilities")) {
        return responseJson({ durable_ack_fields: ["receiver_request_id", "content_hash", "submitted_content_hash", "outcome"] }, { requestId: "capability-1" });
      }
      return responseJson({ error: "unexpected_service_worker_provider_fetch" }, { ok: false, status: 500 });
    });

    const started = await sendRuntimeMessage({
      type: "polylogue.backfill.start",
      provider: "chatgpt",
      cutoff: "2026-01-01T00:00:00Z",
      policy: { baseCadenceMs: 1000 },
    });

    expect(started.ok).toBe(true);
    await vi.waitFor(() => expect(globalThis.chrome.scripting.executeScript).toHaveBeenCalledWith(expect.objectContaining({
      target: { tabId: 42 },
      world: "MAIN",
      func: expect.any(Function),
      args: [expect.objectContaining({ provider: "chatgpt", operation: "inventory" })],
    })));
    // Provider inventory/conversation fetches route through the page bridge
    // (chrome.scripting.executeScript above), never through this
    // service-worker `fetch`. Receiver capability and CaptureJob durability
    // traffic are the only legitimate service-worker fetches for this flow.
    for (const call of fetchCalls) {
      expect(call.url).toMatch(/\/v1\/(browser-captures\/capabilities|capture-jobs)/);
    }
    expect(fetchCalls.filter((call) => String(call.url).includes("/v1/browser-captures/capabilities"))).toHaveLength(1);
    expect(globalThis.chrome.tabs.sendMessage).not.toHaveBeenCalled();
    expect(globalThis.chrome.tabs.create).not.toHaveBeenCalled();
  });

  it("keeps concurrent backfill requests with different cutoffs distinct", async () => {
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      fetchCalls.push({ url, options });
      const captureJobResponse = captureJobFixtureResponse(url, options);
      if (captureJobResponse) return captureJobResponse;
      if (String(url).endsWith("/v1/browser-captures/capabilities")) {
        return responseJson({ durable_ack_fields: ["receiver_request_id", "content_hash", "submitted_content_hash", "outcome"] });
      }
      return responseJson({ error: "unexpected_receiver_request" }, { ok: false, status: 500 });
    });

    const requests = ["2026-01-01T00:00:00Z", "2025-01-01T00:00:00Z"].map((cutoff) => sendRuntimeMessage({
      type: "polylogue.backfill.start",
      provider: "chatgpt",
      cutoff,
      policy: { baseCadenceMs: 1000 },
    }));
    const responses = await Promise.all(requests);
    expect(responses).toHaveLength(2);
    expect(responses.filter((response) => response.ok)).toHaveLength(1);
    expect(responses.find((response) => !response.ok)).toMatchObject({
      ok: false,
      error: expect.stringMatching(/^backfill_job_already_active:chatgpt:/),
    });

    await vi.waitFor(() => expect(globalThis.chrome.scripting.executeScript).toHaveBeenCalled());
    expect(globalThis.chrome.tabs.create).not.toHaveBeenCalled();
    expect(globalThis.chrome.scripting.executeScript.mock.calls.every(([details]) => details.target.tabId === 42)).toBe(true);
  });

  it("coalesces equivalent concurrent backfill requests", async () => {
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      const captureJobResponse = captureJobFixtureResponse(url, options);
      if (captureJobResponse) return captureJobResponse;
      if (String(url).endsWith("/v1/browser-captures/capabilities")) {
        return responseJson({ durable_ack_fields: ["receiver_request_id", "content_hash", "submitted_content_hash", "outcome"] });
      }
      return responseJson({ error: "unexpected_receiver_request" }, { ok: false, status: 500 });
    });

    const requests = [1, 2].map(() => sendRuntimeMessage({
      type: "polylogue.backfill.start",
      provider: "chatgpt",
      cutoff: "2026-01-01T00:00:00Z",
      policy: { baseCadenceMs: 1000 },
    }));
    const responses = await Promise.all(requests);

    expect(responses[0].job.id).toBe(responses[1].job.id);
  });

  it("coalesces equivalent requests despite object insertion order", async () => {
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      const captureJobResponse = captureJobFixtureResponse(url, options);
      if (captureJobResponse) return captureJobResponse;
      if (String(url).endsWith("/v1/browser-captures/capabilities")) {
        return responseJson({ durable_ack_fields: ["receiver_request_id", "content_hash", "submitted_content_hash", "outcome"] });
      }
      return responseJson({ error: "unexpected_receiver_request" }, { ok: false, status: 500 });
    });

    const responses = await Promise.all([
      sendRuntimeMessage({
        type: "polylogue.backfill.start",
        provider: "chatgpt",
        cutoff: "2026-01-01T00:00:00Z",
        policy: { baseCadenceMs: 1000, nested: { first: true, second: false } },
        provider_options: { account: "fixture", region: "test" },
      }),
      sendRuntimeMessage({
        type: "polylogue.backfill.start",
        provider: "chatgpt",
        cutoff: "2026-01-01T00:00:00Z",
        policy: { nested: { second: false, first: true }, baseCadenceMs: 1000 },
        provider_options: { region: "test", account: "fixture" },
      }),
    ]);

    expect(responses[0].job.id).toBe(responses[1].job.id);
  });

  it.each(["lookup", "removal"])("transport cleanup retains original custody after a %s fault", async (fault) => {
    await loadBackground();
    const key = "polylogueProviderTransportTab:chatgpt";
    const alarm = "polylogueBackfillTransportCleanup:chatgpt:99";
    tabs = [{ id: 99, url: "https://chatgpt.com/", active: false, status: "complete" }];
    sessionStored = { [key]: 99 };
    const api = fault === "lookup" ? globalThis.chrome.tabs.get : globalThis.chrome.tabs.remove;
    api.mockRejectedValueOnce(new Error("synthetic_transport_fault"));
    globalThis.chrome.alarms.clear.mockClear();
    alarmListener({ name: alarm });
    await vi.waitFor(() => expect(stored.polylogueDebugLog).toEqual(expect.arrayContaining([
      expect.objectContaining({ stage: "provider_transport_cleanup_pending", tab_id: 99 }),
    ])));
    expect(sessionStored[key]).toBe(99);
    expect(globalThis.chrome.alarms.clear).not.toHaveBeenCalledWith(alarm);
    expect(tabs.some(tab => tab.id === 99)).toBe(true);
    alarmListener({ name: alarm });
    await vi.waitFor(() => expect(sessionStored[key]).toBeUndefined());
    expect(globalThis.chrome.tabs.remove).toHaveBeenCalledWith(99);
    expect(tabs.some(tab => tab.id === 99)).toBe(false);
  });

  it.each([null, 77])("transport cleanup refuses a stale wake after ownership became %s", async (current) => {
    await loadBackground();
    const key = "polylogueProviderTransportTab:chatgpt";
    const alarm = "polylogueBackfillTransportCleanup:chatgpt:99";
    tabs = [{ id: 99, url: "https://chatgpt.com/", active: false, status: "complete" }];
    sessionStored = current === null ? {} : { [key]: current };
    globalThis.chrome.alarms.clear.mockClear();
    alarmListener({ name: alarm });
    await vi.waitFor(() => expect(globalThis.chrome.alarms.clear).toHaveBeenCalledWith(alarm));
    expect(globalThis.chrome.tabs.remove).not.toHaveBeenCalledWith(99);
    expect(sessionStored[key]).toBe(current === null ? undefined : current);
  });

  it("transport cleanup retires custody only after Chrome positively reports the original tab absent", async () => {
    await loadBackground();
    const key = "polylogueProviderTransportTab:chatgpt";
    const alarm = "polylogueBackfillTransportCleanup:chatgpt:99";
    sessionStored = { [key]: 99 };
    globalThis.chrome.tabs.get.mockRejectedValueOnce(new Error("No tab with id: 99."));
    alarmListener({ name: alarm });
    await vi.waitFor(() => expect(sessionStored[key]).toBeUndefined());
    expect(globalThis.chrome.tabs.remove).not.toHaveBeenCalledWith(99);
    expect(globalThis.chrome.alarms.clear).toHaveBeenCalledWith(alarm);
  });

  it("transport cleanup rechecks custody after an in-flight lookup", async () => {
    await loadBackground();
    const key = "polylogueProviderTransportTab:chatgpt";
    const alarm = "polylogueBackfillTransportCleanup:chatgpt:99";
    sessionStored = { [key]: 99 };
    let release;
    globalThis.chrome.tabs.get.mockImplementationOnce(() => new Promise(resolve => { release = resolve; }));
    alarmListener({ name: alarm });
    await vi.waitFor(() => expect(release).toBeTypeOf("function"));
    sessionStored = { [key]: 77 };
    release({ id: 99, url: "https://chatgpt.com/", active: false });
    await vi.waitFor(() => expect(globalThis.chrome.alarms.clear).toHaveBeenCalledWith(alarm));
    expect(globalThis.chrome.tabs.remove).not.toHaveBeenCalledWith(99);
    expect(sessionStored[key]).toBe(77);
  });

  it.each(["existing_lookup", "setup_cleanup"])("transport cleanup preserves original setup failure and custody: %s", async (fault) => {
    await loadBackground({ polylogueReceiverPairing: {
      state: "online", receiver_id: "rx-transport", api_schema: "polylogue-browser-capture/v1",
    } });
    const key = "polylogueProviderTransportTab:chatgpt";
    const alarm = "polylogueBackfillTransportCleanup:chatgpt:99";
    const action = { action_id: "transport-fault", receiver_id: "rx-transport", provider: "chatgpt",
      operation: "conversation.create", target: {}, text: "Neutral transport fixture", attachments: [],
      presentation: {}, submit_policy: "submit_once", status: "leased" };
    const updates = [];
    let claimed = false;
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      if (String(url).endsWith("/v1/status")) return responseJson({ ok: true, receiver_id: "rx-transport", api_schema: "polylogue-browser-capture/v1" });
      if (String(url).includes("/v1/browser-actions?claim_by=")) {
        const actions = claimed ? [] : [action]; claimed = true; return responseJson({ actions });
      }
      if (String(url).endsWith("/events")) { updates.push(JSON.parse(options.body)); return responseJson({ action }); }
      return responseJson({ error: "unexpected" }, { ok: false, status: 500 });
    });
    if (fault === "existing_lookup") sessionStored = { [key]: 99 };
    else {
      globalThis.chrome.tabs.create.mockResolvedValueOnce({ id: 99, url: "https://chatgpt.com/", active: false, status: "loading" });
      globalThis.chrome.tabs.remove.mockRejectedValueOnce(new Error("synthetic_removal_fault"));
    }
    globalThis.chrome.tabs.get.mockRejectedValue(new Error("synthetic_original_lookup_fault"));
    alarmListener({ name: "polylogueBrowserActionWake" });
    await vi.waitFor(() => expect(updates.at(-1)?.phase).toBe("provider_action_failed"));
    expect(sessionStored[key]).toBe(99);
    expect(globalThis.chrome.alarms.clear).not.toHaveBeenCalledWith(alarm);
    if (fault === "existing_lookup") expect(globalThis.chrome.tabs.create).not.toHaveBeenCalled();
    expect(stored.polylogueCaptureLog).toEqual(expect.arrayContaining([
      expect.objectContaining({ reason: "browser_action_failed", error: "synthetic_original_lookup_fault" }),
    ]));
  });

  it("does not replace a provider transport tab once the operator activates it", async () => {
    await loadBackground();
    const transportKey = "polylogueProviderTransportTab:chatgpt";
    const takenKey = "polylogueProviderTransportOperatorTaken:chatgpt";
    tabs = [{ id: 99, url: "https://chatgpt.com/", active: true, status: "complete" }];
    sessionStored = { [transportKey]: 99 };

    alarmListener({ name: "polylogueBackfillTransportCleanup:chatgpt:99" });
    await vi.waitFor(() => expect(sessionStored[takenKey]).toBe(99));
    expect(globalThis.chrome.tabs.remove).not.toHaveBeenCalledWith(99);

    const started = await sendRuntimeMessage({
      type: "polylogue.backfill.start",
      provider: "chatgpt",
      cutoff: "2026-01-01T00:00:00Z",
    });

    expect(started.ok).toBe(true);
    await vi.waitFor(() => expect(globalThis.chrome.tabs.get).toHaveBeenCalledWith(99));
    expect(globalThis.chrome.tabs.create.mock.calls.filter(([request]) => request.url === "https://chatgpt.com/")).toHaveLength(0);

    tabs = [];
    removedListener(99);
    await vi.waitFor(() => expect(sessionStored[transportKey]).toBeUndefined());
    expect(sessionStored[takenKey]).toBeUndefined();
  });

  it("cancels queued provider acquisition without waiting for another provider request or issuing new traffic", async () => {
    const { BackfillCoordinator } = await import("../src/backfill/coordinator.js");
    let adapter;
    const wake = vi.spyOn(BackfillCoordinator.prototype, "wake").mockImplementation(async function () {
      adapter = this.adapters.chatgpt;
    });
    await sendRuntimeMessage({ type: "polylogue.backfill.status" });
    alarmListener({ name: "polylogueBackfillWake:cancel-admission-control" });
    await vi.waitFor(() => expect(adapter).toBeDefined());
    wake.mockRestore();
    let finishFirst; let acquisitions = 0;
    globalThis.chrome.scripting.executeScript = mockPageScript(async (details) => {
      if (details.args?.[0]?.operation !== "inventory") return [{ result: { ok: true, response: {} } }];
      acquisitions += 1;
      await new Promise((resolve) => { finishFirst = resolve; });
      return [{ result: { ok: true, response: { ok: true, status: 200, contentType: "application/json",
        body: JSON.stringify({ items: [], total: 0 }) } } }];
    });
    const first = adapter.enumerate("0", null, new globalThis.AbortController().signal);
    await vi.waitFor(() => expect(acquisitions).toBe(1));
    const controller = new globalThis.AbortController();
    const second = adapter.enumerate("0", null, controller.signal);
    const cancelled = new Error("capture_cancelled"); cancelled.name = "AbortError";
    controller.abort(cancelled);
    await expect(second).rejects.toBe(cancelled);
    expect(acquisitions).toBe(1);
    finishFirst();
    await expect(first).resolves.toMatchObject({ classification: "success" });
    expect(acquisitions).toBe(1);
    const store = new IndexedDbBackfillStore(globalThis.indexedDB);
    const accountScope = await deriveAccountScope("cjs1:fixture-stable-namespace", "chatgpt", "test-account-chatgpt");
    await store.putJob({ id: "identity-recovery", provider: "chatgpt", account_scope: accountScope, status: "running",
      execution_owner: "identity-owner", execution_generation: 1 });
    const staging = new CaptureStaging(globalThis.navigator.storage, store);
    const owner = { tab_id: 42, document_id: "identity-document", provider: "chatgpt" };
    const ref = await staging.begin(owner, { kind: "native-response",
      source_url: "https://chatgpt.com/backend-api/conversation/restored-session",
      queue_context: { jobId: "identity-recovery", itemId: "identity-item" } });
    await staging.append(ref, owner, 0, globalThis.Buffer.from(JSON.stringify({ id: "unrelated-metadata-id", conversation_id: "restored-session", mapping: {} })).toString("base64"));
    await staging.seal(ref, owner);
    const item = { id: "identity-item", job_id: "identity-recovery", native_id: "restored-session", raw_acquisition_ref: ref, state: "eligible" };
    await store.putQueue(item);
    const restored = await adapter.fetchNative("restored-session", new globalThis.AbortController().signal,
      { item, jobId: "identity-recovery", owner: "identity-owner", generation: 1 });
    expect(restored.captureRawRef).toEqual(ref);
    expect(await restored.json()).toMatchObject({ conversation_id: "restored-session", id: "unrelated-metadata-id" });
    expect(acquisitions).toBe(1);
  });

  it("retries coordinator initialization after recovery storage fails once", async () => {
    const get = globalThis.chrome.storage.local.get;
    let failed = false;
    globalThis.chrome.storage.local.get = vi.fn(async (defaults) => {
      if (!failed && Object.hasOwn(defaults, "polylogueBackfillRecoveryCheckpoint")) {
        failed = true;
        throw new Error("synthetic_recovery_storage_failure");
      }
      return get(defaults);
    });
    expect(await sendRuntimeMessage({ type: "polylogue.backfill.status" })).toMatchObject({ ok: false, error: "synthetic_recovery_storage_failure" });
    expect(await sendRuntimeMessage({ type: "polylogue.backfill.status" })).toMatchObject({ ok: true, jobs: [] });
  });

  it("admits receiver capability preflight without an invented request deadline", async () => {
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      fetchCalls.push({ url, options });
      const captureJobResponse = captureJobFixtureResponse(url, options);
      if (captureJobResponse) return captureJobResponse;
      return responseJson({ durable_ack_fields: ["receiver_request_id", "content_hash", "submitted_content_hash", "outcome"] }, { requestId: "capability-1" });
    });
    await sendRuntimeMessage({ type: "polylogue.backfill.start", provider: "chatgpt", cutoff: "2026-01-01T00:00:00Z" });
    await vi.waitFor(() => expect(fetchCalls.some((call) => String(call.url).includes("/v1/browser-captures/capabilities"))).toBe(true));
    const capabilityCall = fetchCalls.find((call) => String(call.url).includes("/v1/browser-captures/capabilities"));
    expect(capabilityCall.options.signal).toBeUndefined();
  });

  it("classifies a reachable receiver missing the capability route as contract-incompatible", async () => {
    globalThis.fetch = vi.fn(async (url, options = {}) => captureJobFixtureResponse(url, options)
      || responseJson({ error: "not_found" }, { ok: false, status: 404 }));
    const started = await sendRuntimeMessage({ type: "polylogue.backfill.start", provider: "chatgpt", cutoff: "2026-01-01T00:00:00Z" });
    // Name the refusal: reaching for `started.job.id` on a rejected start
    // reports "undefined has no 'id'" and buries the reason.
    expect(started).toMatchObject({ ok: true });
    expect(started.job).toMatchObject({ status: "paused", cooldown_reason: "receiver_contract_incompatible" });
    expect(globalThis.chrome.scripting.executeScript.mock.calls.filter(
      ([details]) => typeof details.args?.[0]?.operation === "string" && details.args[0].operation !== "identity",
    )).toHaveLength(0);
  });

  it("restores a packaged recovery checkpoint as an actionable paused job and ignores its alarm", async () => {
    await loadBackground({
      polylogueBackfillRecoveryCheckpoint: {
        version: 1,
        jobs: [{
          id: "recovered-job", provider: "chatgpt", cutoff: "2026-01-01T00:00:00Z", status: "running",
          inventory_cursor: "17", policy: { leaseMs: 180000, maxDailyRequests: 10 }, execution_generation: 0,
          learned_cadence_ms: 40000, daily_requests: 7, last_ack: { receiver_request_id: "ack-1", content_hash: "hash-1" },
        }],
        queue: [{ id: "recovered-item", job_id: "recovered-job", provider: "chatgpt", native_id: "one", state: "captured_waiting_receiver", content_hash: "hash-1" }],
        revisions: [],
      },
    });
    const status = await sendRuntimeMessage({ type: "polylogue.backfill.status" });
    expect(status.jobs[0]).toMatchObject({
      id: "recovered-job", status: "paused", cooldown_reason: "browser_profile_recovery_required",
      inventory_cursor: "17", daily_requests: 7, last_ack: { receiver_request_id: "ack-1" },
      progress: { operator_action: 1 },
    });
    const pageWorkBeforeAlarm = globalThis.chrome.scripting.executeScript.mock.calls.filter(
      ([details]) => typeof details.args?.[0]?.operation === "string" && details.args[0].operation !== "identity",
    ).length;
    alarmListener({ name: "polylogueBackfillWake:recovered-job" });
    await Promise.resolve();
    expect(globalThis.chrome.scripting.executeScript.mock.calls.filter(
      ([details]) => typeof details.args?.[0]?.operation === "string" && details.args[0].operation !== "identity",
    )).toHaveLength(pageWorkBeforeAlarm);
  });

  it("commits the immutable checkpoint artifact through CaptureJobs without recreating the retired local ledger", async () => {
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      fetchCalls.push({ url, options });
      const captureJobResponse = captureJobFixtureResponse(url, options);
      if (captureJobResponse) return captureJobResponse;
      if (String(url).endsWith("/v1/browser-captures/capabilities")) {
        return responseJson({ durable_ack_fields: ["receiver_request_id", "content_hash", "submitted_content_hash", "outcome"] });
      }
      return responseJson({ error: "unexpected_receiver_request" }, { ok: false, status: 500 });
    });

    await sendRuntimeMessage({ type: "polylogue.backfill.start", provider: "chatgpt", cutoff: "2026-01-01T00:00:00Z" });

    expect(fetchCalls.some((call) => new globalThis.URL(call.url).pathname === "/v1/backfill-checkpoint")).toBe(false);
    const captureJobCommit = fetchCalls.findIndex((call) => new globalThis.URL(call.url).pathname.endsWith("/checkpoint"));
    expect(captureJobCommit).toBeGreaterThanOrEqual(0);
    const uploaded = fetchCalls[captureJobCommit].options;
    const descriptor = captureJobRequestBody(uploaded);
    const bytes = await uploaded.body.text();
    expect(descriptor.digest).toBe(`sha256:${createHash("sha256").update(bytes).digest("hex")}`);
    expect(JSON.parse(bytes).jobs).toEqual([expect.objectContaining({ provider: "chatgpt", status: "running" })]);
    expect(stored.polylogueBackfillRecoveryCheckpoint).toBeUndefined();
    expect(globalThis.chrome.storage.local.set.mock.calls.some(([patch]) => "polylogueBackfillRecoveryCheckpoint" in patch)).toBe(false);
  });

  it("derives CaptureJob scope from live provider identity without disclosing the handle", async () => {
    const accountHandle = "stable-chatgpt-account-id";
    globalThis.chrome.scripting.executeScript = mockPageScript(async (details) => {
      const request = details.args?.[0];
      if (request?.operation === "identity") {
        return [{ result: { ok: true, response: { accountHandle } } }];
      }
      const body = request?.operation === "inventory"
        ? { items: [], total: 0 }
        : { id: "backfill-1", mapping: { node: { id: "node", parent: null, message: { id: "message", author: { role: "user" }, content: { content_type: "text", parts: ["synthetic message"] } } } } };
      return [{ result: { ok: true, response: {
        ok: true,
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(body),
      } } }];
    });
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      fetchCalls.push({ url, options });
      const artifactResponse = checkpointArtifactFixtureResponse(url);
      if (artifactResponse) return artifactResponse;
      const path = new globalThis.URL(url).pathname;
      if (path === "/v1/capture-jobs/capabilities") {
        return responseJson({
          schema: "polylogue.capture-jobs.capabilities.v1", checkpoint_transport: "canonical-artifact-v1",
          protocol_min: 2,
          protocol_max: 2,
          scope_namespace: "cjs1:identity-test-namespace",
        });
      }
      if (path === "/v1/browser-captures/capabilities") {
        return responseJson({ durable_ack_fields: ["receiver_request_id", "content_hash", "submitted_content_hash", "outcome"] });
      }
      if (path === "/v1/capture-jobs/discover") return responseJson({ jobs: [] });
      if (path === "/v1/capture-jobs") {
        return responseJson({ job: {
          job_id: "capture-job-1", provider: "chatgpt", revision: 0, lease_generation: 0,
        } }, { status: 201 });
      }
      if (path.endsWith("/adopt")) {
        return responseJson({
          job: { job_id: "capture-job-1", provider: "chatgpt", revision: 1, lease_generation: 1 },
          lease: { lease_id: "lease-1", generation: 1, proof: "proof-1" },
        });
      }
      if (path.endsWith("/update")) {
        return responseJson({
          job: {
            job_id: "capture-job-1", provider: "chatgpt", revision: 2, lease_generation: 1,
            lease_expires_at: "2026-07-16T10:02:00Z",
          },
          receipt: { kind: "capture_job_update" },
        });
      }
      if (path.endsWith("/checkpoint")) {
        return checkpointFixtureResponse({ job_id: "capture-job-1", revision: 3 }, options);
      }
      return responseJson({ error: "unexpected_receiver_request" }, { ok: false, status: 500 });
    });

    await sendRuntimeMessage({
      type: "polylogue.backfill.start",
      provider: "chatgpt",
      cutoff: "2026-01-01T00:00:00Z",
    });

    await vi.waitFor(() => expect(fetchCalls.some(
      (call) => new globalThis.URL(call.url).pathname.endsWith("/checkpoint") && call.options.method === "PUT",
    )).toBe(true));
    const captureJobCalls = fetchCalls.filter((call) => new globalThis.URL(call.url).pathname.startsWith("/v1/capture-jobs"));
    expect(captureJobCalls.length).toBeGreaterThanOrEqual(5);
    for (const call of captureJobCalls) {
      if (!call.options.body) continue;
      const encoded = typeof call.options.body === "string" ? call.options.body : await call.options.body.text();
      expect(encoded).not.toContain(accountHandle);
      expect(encoded).not.toContain("paired:");
      const body = captureJobRequestBody(call.options);
      if (body.scope?.kind === "account") expect(body.scope.key).toMatch(/^h1:/);
    }
    expect(JSON.stringify(stored)).not.toContain(accountHandle);
  });

  it("refuses a successful but malformed identity response before creating a receiver job", async () => {
    globalThis.chrome.scripting.executeScript = mockPageScript(async () => [{ result: { ok: true, response: {} } }], { accountHandle: null });
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      fetchCalls.push({ url, options });
      return responseJson({ error: "unexpected_receiver_request" }, { ok: false, status: 500 });
    });
    const result = await sendRuntimeMessage({ type: "polylogue.backfill.start", provider: "chatgpt", cutoff: "2026-01-01T00:00:00Z" });
    expect(result.ok).toBe(false);
    expect(fetchCalls.some((call) => new globalThis.URL(call.url).pathname === "/v1/capture-jobs")).toBe(false);
    expect((await new IndexedDbBackfillStore(globalThis.indexedDB).jobPage()).jobs).toEqual([]);
  });

  it("fails CaptureJob publication closed when provider identity is unavailable", async () => {
    globalThis.chrome.scripting.executeScript = mockPageScript(async (details) => {
      const request = details.args?.[0];
      if (request?.operation === "identity") {
        return [{ result: { ok: false, error: "backfill_bridge_auth_context_unavailable" } }];
      }
      return [{ result: { ok: true, response: {
        ok: true,
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ items: [], total: 0 }),
      } } }];
    }, { accountHandle: null });
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      fetchCalls.push({ url, options });
      const artifactResponse = checkpointArtifactFixtureResponse(url);
      if (artifactResponse) return artifactResponse;
      const path = new globalThis.URL(url).pathname;
      if (path === "/v1/capture-jobs/capabilities") {
        return responseJson({
          schema: "polylogue.capture-jobs.capabilities.v1", checkpoint_transport: "canonical-artifact-v1",
          protocol_min: 2,
          protocol_max: 2,
          scope_namespace: "cjs1:profile-recovery-namespace",
        });
      }
      if (path === "/v1/browser-captures/capabilities") {
        return responseJson({ durable_ack_fields: ["receiver_request_id", "content_hash", "submitted_content_hash", "outcome"] });
      }
      return responseJson({ error: "unexpected_receiver_request" }, { ok: false, status: 500 });
    });

    await sendRuntimeMessage({
      type: "polylogue.backfill.start",
      provider: "chatgpt",
      cutoff: "2026-01-01T00:00:00Z",
    });

    await vi.waitFor(() => expect(globalThis.chrome.scripting.executeScript).toHaveBeenCalledWith(
      expect.objectContaining({ args: [{ provider: "chatgpt", operation: "identity", params: {}, ownerId: globalThis.chrome.runtime.id }] }),
    ));
    expect(fetchCalls.some((call) => new globalThis.URL(call.url).pathname.startsWith("/v1/capture-jobs"))).toBe(false);
    expect(JSON.stringify(stored)).not.toContain("paired:chatgpt");
  });

  it("delivers the complete staged ChatGPT mapping to the receiver", async () => {
    const pairing = { state: "online", receiver_id: "rx-chunk-test", api_schema: "polylogue-browser-capture/v1", endpoint: "http://127.0.0.1:8875" };
    stored.polylogueReceiverPairing = pairing;
    tabs = [{ id: 42, url: "https://chatgpt.com/", title: "ChatGPT" }];
    globalThis.chrome.tabs.sendMessage = vi.fn(async (_tabId, message) => message.type === "polylogue.acquireRecordAssets" ? { ok: true, acquisition: null } : { ok: false, error: "unexpected_capture_message" });
    const accountHandle = "stable-chatgpt-account-id";
    const conversationBody = {
      id: "chunked-conversation", title: "Native",
      mapping: {
        "node-a": { id: "node-a", parent: null, message: { id: "message-a", author: { role: "user" }, content: { content_type: "text", parts: ["first half"] } } },
        "node-b": { id: "node-b", parent: "node-a", message: { id: "message-b", author: { role: "assistant" }, content: { content_type: "text", parts: ["second half"] } } },
      },
    };
    globalThis.chrome.scripting.executeScript = mockPageScript(async (details) => {
      const request = details.args?.[0];
      if (request?.operation === "identity") return [{ result: { ok: true, response: { accountHandle } } }];
      if (request?.operation === "inventory") {
        return [{ result: { ok: true, response: {
          ok: true, status: 200, contentType: "application/json",
          body: JSON.stringify({ items: [{ id: "chunked-conversation" }], total: 1 }),
        } } }];
      }
      if (request?.operation === "conversation") {
        return [{ result: { ok: true, response: {
          ok: true, status: 200, contentType: "application/json",
          body: JSON.stringify(conversationBody),
        } } }];
      }
      // Other passive features (e.g. ambient DOM reconciliation for the open
      // tab) also drive chrome.scripting.executeScript; this fixture only
      // cares about the ChatGPT backfill bridge above.
      return [{ result: undefined }];
    });
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      fetchCalls.push({ url, options });
      const captureJobResponse = captureJobFixtureResponse(url, options);
      if (captureJobResponse) return captureJobResponse;
      const artifactResponse = checkpointArtifactFixtureResponse(url);
      if (artifactResponse) return artifactResponse;
      const path = new globalThis.URL(url).pathname;
      if (path === "/v1/status") {
        return responseJson({ ok: true, receiver_id: pairing.receiver_id, api_schema: pairing.api_schema });
      }
      if (path === "/v1/browser-captures/capabilities") {
        return responseJson({ durable_ack_fields: ["receiver_request_id", "content_hash", "submitted_content_hash", "outcome"] });
      }
      if (path === "/v1/browser-captures") {
        const digest = await globalThis.crypto.subtle.digest("SHA-256", await options.body.arrayBuffer());
        const contentHash = [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
        return responseJson({ ok: true, provider: "chatgpt", provider_session_id: "chunked-conversation", state: "complete", artifact_ref: "chatgpt/chunked-conversation.json", outcome: "accepted", submitted_content_hash: contentHash, content_hash: contentHash });
      }
      return responseJson({ error: "unexpected_receiver_request" }, { ok: false, status: 500 });
    });

    // The coordinator real-time paces successive provider requests
    // (learned_cadence_ms, ~15s) to avoid hammering the ChatGPT API, and one
    // wake only ever enumerates a single inventory partition or processes
    // queued items, never both. Advance the coordinator's injected clock
    // (BackfillCoordinator defaults to `() => Date.now()`) instead of really
    // sleeping ~75s of wall-clock time per test run.
    let simulatedNowMs = Date.now();
    const clockSpy = vi.spyOn(Date, "now").mockImplementation(() => simulatedNowMs);
    try {
      const started = await sendRuntimeMessage({ type: "polylogue.backfill.start", provider: "chatgpt", cutoff: "2026-01-01T00:00:00Z" });
      // Name the refusal: reaching for `started.job.id` on a rejected start
      // reports "undefined has no 'id'" and buries the reason.
      expect(started).toMatchObject({ ok: true });
      await vi.waitFor(() => expect(globalThis.chrome.scripting.executeScript).toHaveBeenCalled());
      // Four ChatGPT inventory partitions (active/archived x starred/
      // unstarred) to walk through before inventory_complete, plus one more
      // wake to fetch the queued conversation.
      for (let wake = 0; wake < 5; wake += 1) {
        simulatedNowMs += 20000;
        alarmListener({ name: `polylogueBackfillWake:${started.job.id}` });
        // Let the wake's async chain (provider fetch, capture, receiver
        // POST, CaptureJob publication) fully settle before advancing the
        // clock again. A later wake may have nothing left to do once the
        // item is already captured, so this does not assert growth.
        await new Promise((resolve) => globalThis.setTimeout(resolve, 150));
        if (fetchCalls.some((call) => new globalThis.URL(call.url).pathname.endsWith("/native/publish") && call.options.method === "POST")) break;
      }
      // The four partitions all resolve to the same single native id, so the
      // Submitting the reassembled capture to the receiver is its own wake
      // step (acquireNextLease for "captured_waiting_receiver" happens
      // before the provider-fetch loop in runLeasedJob), so it can still be
      // pending after the wake that only just finished the provider fetch.
      for (let submitWake = 0; submitWake < 3; submitWake += 1) {
        if (fetchCalls.some((call) => new globalThis.URL(call.url).pathname.endsWith("/native/publish") && call.options.method === "POST")) break;
        simulatedNowMs += 20000;
        alarmListener({ name: `polylogueBackfillWake:${started.job.id}` });
        await new Promise((resolve) => globalThis.setTimeout(resolve, 150));
      }
      await vi.waitFor(() => expect(fetchCalls.some((call) => new globalThis.URL(call.url).pathname.endsWith("/native/publish") && call.options.method === "POST")).toBe(true), { timeout: 4000 });
    } finally {
      clockSpy.mockRestore();
    }

    const memberCall = fetchCalls.find((call) => new globalThis.URL(call.url).pathname.endsWith("/native/member"));
    expect(JSON.parse(await memberCall.options.body.text())).toEqual(conversationBody);
    const publication = fetchCalls.find((call) => new globalThis.URL(call.url).pathname.endsWith("/native/publish"));
    expect(JSON.parse(publication.options.body).sha256).toBe("b".repeat(64));
  }, 20000);

  it("retains thoughts literally through native member publication", async () => {
    const pairing = { state: "online", receiver_id: "rx-thoughts-fallback-test", api_schema: "polylogue-browser-capture/v1", endpoint: "http://127.0.0.1:8875" };
    stored.polylogueReceiverPairing = pairing;
    tabs = [{ id: 42, url: "https://chatgpt.com/", title: "ChatGPT" }];
    globalThis.chrome.tabs.sendMessage = vi.fn(async (_tabId, message) => message.type === "polylogue.acquireRecordAssets" ? { ok: true, acquisition: null } : { ok: false, error: "unexpected_capture_message" });

    const accountHandle = "stable-chatgpt-account-id";
    const conversationBody = {
      id: "thoughts-only-conversation", title: "Reasoning only",
      mapping: {
        reasoning: {
          id: "reasoning", parent: null,
          message: {
            id: "reasoning-msg", author: { role: "assistant" },
            content: { content_type: "thoughts", thoughts: [{ summary: "Weighing options", content: "Considered A and B, chose A." }] },
          },
        },
      },
    };
    globalThis.chrome.scripting.executeScript = mockPageScript(async (details) => {
      const request = details.args?.[0];
      if (request?.operation === "identity") return [{ result: { ok: true, response: { accountHandle } } }];
      if (request?.operation === "inventory") {
        return [{ result: { ok: true, response: {
          ok: true, status: 200, contentType: "application/json",
          body: JSON.stringify({ items: [{ id: "thoughts-only-conversation" }], total: 1 }),
        } } }];
      }
      if (request?.operation === "conversation") {
        return [{ result: { ok: true, response: {
          ok: true, status: 200, contentType: "application/json",
          body: JSON.stringify(conversationBody),
        } } }];
      }
      return [{ result: undefined }];
    });
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      fetchCalls.push({ url, options });
      const captureJobResponse = captureJobFixtureResponse(url, options);
      if (captureJobResponse) return captureJobResponse;
      const artifactResponse = checkpointArtifactFixtureResponse(url);
      if (artifactResponse) return artifactResponse;
      const path = new globalThis.URL(url).pathname;
      if (path === "/v1/status") {
        return responseJson({ ok: true, receiver_id: pairing.receiver_id, api_schema: pairing.api_schema });
      }
      if (path === "/v1/browser-captures/capabilities") {
        return responseJson({ durable_ack_fields: ["receiver_request_id", "content_hash", "submitted_content_hash", "outcome"] });
      }
      if (path === "/v1/browser-captures") {
        const digest = await globalThis.crypto.subtle.digest("SHA-256", await options.body.arrayBuffer());
        const contentHash = [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
        return responseJson({ ok: true, provider: "chatgpt", provider_session_id: "thoughts-only-conversation", state: "complete", artifact_ref: "chatgpt/thoughts-only-conversation.json", content_hash: contentHash });
      }
      return responseJson({ error: "unexpected_receiver_request" }, { ok: false, status: 500 });
    });

    let simulatedNowMs = Date.now();
    const clockSpy = vi.spyOn(Date, "now").mockImplementation(() => simulatedNowMs);
    try {
      const recovered = await sendRuntimeMessage({ type: "polylogue.backfill.status" });
      expect(recovered, `backfill recovery failed: ${recovered.error || "unknown"}`).toMatchObject({ ok: true });
      for (const activeJob of recovered.jobs.filter((job) => ["running", "paused"].includes(job.status))) {
        const cancelled = await sendRuntimeMessage({ type: "polylogue.backfill.control", job_id: activeJob.id, action: "cancel" });
        expect(cancelled, `backfill cleanup refused: ${cancelled.error || "unknown"}`).toMatchObject({ ok: true, job: { status: "cancelled" } });
      }
      const started = await sendRuntimeMessage({ type: "polylogue.backfill.start", provider: "chatgpt", cutoff: "2026-01-01T00:00:00Z" });
      expect(started, `backfill start refused: ${started.error || "unknown"}`).toMatchObject({ ok: true });
      await vi.waitFor(() => expect(globalThis.chrome.scripting.executeScript).toHaveBeenCalled());
      const capturePosted = () =>
        fetchCalls.some((call) => new globalThis.URL(call.url).pathname.endsWith("/native/publish") && call.options.method === "POST");
      // The coordinator advances one step per wake, so wakes -- not elapsed
      // real time -- are what drive it forward. Wake it again as soon as the
      // previous wake has released the job's execution lease, and never while
      // that lease is held: a wake arriving mid-execution acquires nothing,
      // and the simulated clock it advances can outrun the lease of the work
      // still in flight. Progress, not a wake count, ends the loop, so a host
      // slow enough to need longer per wake gets longer and one that needs
      // more wakes gets more; this test's own timeout is the only bound.
      const jobSnapshot = async () => {
        const status = await new Promise((resolve) => {
          messageListener({ type: "polylogue.backfill.status" }, {}, resolve);
        });
        return status.jobs.find((entry) => entry.id === started.job.id) || null;
      };
      let job = await jobSnapshot();
      let wokenGeneration = -1;
      while (!capturePosted() && job?.status === "running") {
        if (!job.execution_owner && job.execution_generation !== wokenGeneration) {
          wokenGeneration = job.execution_generation;
          simulatedNowMs += 20000;
          alarmListener({ name: `polylogueBackfillWake:${started.job.id}` });
        }
        await new Promise((resolve) => globalThis.setTimeout(resolve, 100));
        job = await jobSnapshot();
      }
      expect(
        capturePosted(),
        `capture never posted; job ${job?.status || "gone"}: ${job?.last_error || job?.cooldown_reason || "no error"}`,
      ).toBe(true);
    } finally {
      clockSpy.mockRestore();
    }

    const memberCall = fetchCalls.find((call) => new globalThis.URL(call.url).pathname.endsWith("/native/member"));
    expect(JSON.parse(await memberCall.options.body.text())).toEqual(conversationBody);
    const publication = fetchCalls.find((call) => new globalThis.URL(call.url).pathname.endsWith("/native/publish"));
    expect(JSON.parse(publication.options.body).sha256).toBe("b".repeat(64));
  }, 20000);

  it("holds the job visibly when the receiver authority cannot commit", async () => {
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      fetchCalls.push({ url, options });
      if (!new globalThis.URL(url).pathname.endsWith("/checkpoint")) {
        const admitted = captureJobFixtureResponse(url, options);
        if (admitted) return admitted;
      }
      if (String(url).endsWith("/v1/browser-captures/capabilities")) {
        return responseJson({ durable_ack_fields: ["receiver_request_id", "content_hash", "submitted_content_hash", "outcome"] });
      }
      return responseJson({ error: "unexpected_receiver_request" }, { ok: false, status: 500 });
    });

    const started = await sendRuntimeMessage({
      type: "polylogue.backfill.start",
      provider: "chatgpt",
      cutoff: "2026-01-01T00:00:00Z",
    });

    expect(started.job.recovery_checkpoint_error).toBe("capture_job_request_failed");
    expect(started.job).toMatchObject({
      status: "paused",
      cooldown_reason: "receiver_capture_job_authority_unavailable",
    });
    expect(stored.polylogueBackfillRecoveryCheckpoint).toBeUndefined();
  });

  it("does not adopt accountless legacy checkpoint evidence after profile loss", async () => {
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      fetchCalls.push({ url, options });
      const captureJobResponse = captureJobFixtureResponse(url, options);
      if (captureJobResponse) return captureJobResponse;
      return responseJson({ error: "unexpected_receiver_request" }, { ok: false, status: 500 });
    });
    const status = await sendRuntimeMessage({ type: "polylogue.backfill.status" });
    expect(status.jobs).toEqual([]);
    expect(fetchCalls.some((call) => new globalThis.URL(call.url).pathname === "/v1/backfill-checkpoint")).toBe(false);
  });

  it.each(["missing_tab", "receiver_refusal"])("retries cached recovery after %s and joins concurrent observers", async (failure) => {
    const existingTabs = tabs;
    if (failure === "missing_tab") tabs = [];
    const accountHandle = "neutral-recovery-account";
    const checkpoint = { version: 1, jobs: [{
      id: "recovered-local-job", provider: "chatgpt", cutoff: "2026-01-01T00:00:00Z",
      status: "complete", inventory_cursor: "done", inventory_complete: true,
      policy: { leaseMs: 180000, maxDailyRequests: 10 }, execution_generation: 0,
      learned_cadence_ms: 40000, daily_requests: 1, last_ack: null,
    }], queue: [], revisions: [] };
    let providerWorkCalls = 0;
    globalThis.chrome.scripting.executeScript = mockPageScript(async (details) => {
      if (details.args?.[0]?.operation === "identity") return [{ result: { ok: true, response: { accountHandle } } }];
      providerWorkCalls += 1;
      throw new Error("unexpected_provider_work");
    });
    // Settle initialization with no provider document, then test the concrete
    // receiver refusal on the separate recovery route with its document present.
    tabs = [];
    await sendRuntimeMessage({ type: "polylogue.backfill.status" });
    tabs = failure === "missing_tab" ? [] : existingTabs;
    let discoveryCalls = 0;
    let releaseDiscovery;
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      fetchCalls.push({ url, options });
      const path = new globalThis.URL(url).pathname;
      const artifact = checkpointArtifactFixtureResponse(url);
      if (artifact) return artifact;
      if (path === "/v1/capture-jobs/capabilities") return captureJobFixtureResponse(url, options);
      if (path === "/v1/capture-jobs/discover") {
        const body = JSON.parse(options.body);
        // Intent-specific checkpoint publication is a separate existing call;
        // suspend only the original account-wide recovery discovery.
        if (body.intent_key) return captureJobFixtureResponse(url, options);
        if (body.provider !== "chatgpt") return responseJson({ jobs: [] });
        discoveryCalls += 1;
        if (failure === "receiver_refusal" && discoveryCalls === 1) return responseJson({ error: "unavailable" }, { ok: false, status: 503 });
        await new Promise(resolve => { releaseDiscovery = resolve; });
        return responseJson({ jobs: [{
          job_id: "receiver-recovered-job", provider: "chatgpt", scope: body.scope,
          intent_key: "recovered-intent", revision: 4, lease_generation: 1,
          updated_at: "2026-07-16T10:00:00Z", checkpoint: checkpointFixture(checkpoint),
        }] });
      }
      if (path === "/v1/capture-jobs/receiver-recovered-job/adopt") return responseJson({
        job: { job_id: "receiver-recovered-job", provider: "chatgpt", intent_key: "recovered-intent",
          revision: 5, lease_generation: 2, checkpoint: checkpointFixture(checkpoint) },
        lease: { lease_id: "recovery-lease", generation: 2, proof: "recovery-proof" },
      });
      return captureJobFixtureResponse(url, options) || responseJson({ error: "unexpected_receiver_request" }, { ok: false, status: 500 });
    });
    expect(await sendRuntimeMessage({ type: "polylogue.backfill.status" })).toMatchObject({ ok: true, jobs: [] });
    // The first post-initialization retry refuses; the next status owns the suspended retry.
    expect(discoveryCalls).toBe(failure === "missing_tab" ? 0 : 1);
    tabs = existingTabs;
    const first = sendRuntimeMessage({ type: "polylogue.backfill.status" });
    await vi.waitFor(() => expect(releaseDiscovery).toBeTypeOf("function"));
    const second = sendRuntimeMessage({ type: "polylogue.backfill.status" });
    await new Promise(resolve => globalThis.setTimeout(resolve, 0));
    expect(discoveryCalls).toBe(failure === "missing_tab" ? 1 : 2);
    releaseDiscovery();
    for (const status of await Promise.all([first, second])) {
      expect(status).toMatchObject({ ok: true, jobs: [expect.objectContaining({
        id: "recovered-local-job", status: "complete", inventory_cursor: "done",
      })] });
    }
    await sendRuntimeMessage({ type: "polylogue.backfill.status" });
    expect(discoveryCalls).toBe(failure === "missing_tab" ? 1 : 2);
    expect(providerWorkCalls).toBe(0);
    const scopedCalls = fetchCalls.filter(call => new globalThis.URL(call.url).pathname === "/v1/capture-jobs/discover");
    expect(scopedCalls.every(call => !call.options.body.includes(accountHandle))).toBe(true);
    expect(JSON.parse(scopedCalls.at(-1).options.body).scope).toMatchObject({ kind: "account", key: expect.stringMatching(/^h1:/) });
  });

  it("retries exact-scope receiver recovery after a provider tab appears", async () => {
    const existingTabs = tabs;
    tabs = [];
    const accountHandle = "stable-account-after-profile-loss";
    const remoteCheckpoint = {
      version: 1,
      jobs: [{
        id: "capture-job-local-work", provider: "chatgpt", cutoff: "2026-01-01T00:00:00Z", status: "running",
        inventory_cursor: "9", policy: { leaseMs: 180000, maxDailyRequests: 10 }, execution_generation: 2,
        learned_cadence_ms: 40000, daily_requests: 3, last_ack: null,
      }],
      queue: [],
      revisions: [],
    };
    let providerWorkCalls = 0;
    globalThis.chrome.scripting.executeScript = mockPageScript(async (details) => {
      const request = details.args?.[0];
      if (request?.operation === "identity") {
        return [{ result: { ok: true, response: { accountHandle } } }];
      }
      providerWorkCalls += 1;
      return [{ result: { ok: true, response: {
        ok: true, status: 200, contentType: "application/json", body: JSON.stringify({ items: [], total: 0 }),
      } } }];
    });
    let adoptAttempts = 0;
    let receiverScope = null;
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      fetchCalls.push({ url, options });
      const artifactResponse = checkpointArtifactFixtureResponse(url);
      if (artifactResponse) return artifactResponse;
      const path = new globalThis.URL(url).pathname;
      if (path === "/v1/capture-jobs/capabilities") {
        return responseJson({
          schema: "polylogue.capture-jobs.capabilities.v1", checkpoint_transport: "canonical-artifact-v1",
          protocol_min: 2,
          protocol_max: 2,
          scope_namespace: "cjs1:new-profile-recovery-namespace",
        });
      }
      if (path === "/v1/capture-jobs/discover") {
        const body = captureJobRequestBody(options);
        if (body.provider !== "chatgpt") return responseJson({ jobs: [] });
        receiverScope = body.scope;
        return responseJson({ jobs: [{
          job_id: "receiver-capture-job",
          provider: "chatgpt",
          scope: receiverScope,
          intent_key: "receiver-intent",
          revision: 4,
          lease_generation: 1,
          updated_at: "2026-07-16T10:00:00Z",
          checkpoint: checkpointFixture(remoteCheckpoint),
        }] });
      }
      if (path.endsWith("/adopt")) {
        adoptAttempts += 1;
        if (adoptAttempts <= 2) {
          return responseJson({ error: { code: "lease_held" } }, { ok: false, status: 409 });
        }
        return responseJson({
          job: {
            job_id: "receiver-capture-job", provider: "chatgpt", scope: receiverScope, intent_key: "receiver-intent",
            revision: 5, lease_generation: 2, checkpoint: checkpointFixture(remoteCheckpoint),
          },
          lease: { lease_id: "recovery-lease", generation: 2, proof: "recovery-proof" },
        });
      }
      if (path.endsWith("/update")) {
        return responseJson({
          job: {
            job_id: "receiver-capture-job", provider: "chatgpt", scope: receiverScope, intent_key: "receiver-intent",
            revision: 6, lease_generation: 2, lease_expires_at: "2099-01-01T00:00:00Z",
            checkpoint_sequence: 1, checkpoint: checkpointFixture(remoteCheckpoint),
          },
          receipt: { kind: "capture_job_update", revision: 6 },
        });
      }
      if (path.endsWith("/checkpoint")) {
        return checkpointFixtureResponse({ job_id: "receiver-capture-job", revision: 7 }, options);
      }
      return responseJson({ error: "unexpected_receiver_request" }, { ok: false, status: 500 });
    });

    expect(await sendRuntimeMessage({ type: "polylogue.backfill.status" })).toMatchObject({ ok: true, jobs: [] });
    tabs = existingTabs;
    // The scoped artifact remains inaccessible while another lease is held.
    // Two explicit observations precede the fixture's lease release.
    expect(await sendRuntimeMessage({ type: "polylogue.backfill.status" })).toMatchObject({ ok: true, jobs: [] });
    expect(await sendRuntimeMessage({ type: "polylogue.backfill.status" })).toMatchObject({ ok: true, jobs: [] });
    expect(adoptAttempts).toBe(2);
    const status = await sendRuntimeMessage({ type: "polylogue.backfill.status" });

    expect(status.jobs[0]).toMatchObject({
      id: "capture-job-local-work",
      status: "paused",
      cooldown_reason: "browser_profile_recovery_required",
      inventory_cursor: "9",
    });
    const discoverCall = fetchCalls.find((call) => new globalThis.URL(call.url).pathname === "/v1/capture-jobs/discover");
    expect(discoverCall).toBeDefined();
    expect(discoverCall.options.body).not.toContain(accountHandle);
    expect(JSON.parse(discoverCall.options.body).scope).toMatchObject({ kind: "account", key: expect.stringMatching(/^h1:/) });
    expect(stored.polylogueExtensionInstanceId).toBeDefined();
    expect(JSON.stringify(stored)).not.toContain(accountHandle);
    expect(fetchCalls.some((call) => new globalThis.URL(call.url).pathname.endsWith("/adopt"))).toBe(true);
    // Recovery adopts once after release; current paused checkpoint publication
    // independently obtains its own verified lease.
    expect(adoptAttempts).toBe(4);
    expect(providerWorkCalls).toBe(0);
    expect(fetchCalls.filter((call) => new globalThis.URL(call.url).pathname.includes("/checkpoint-artifacts/")).length).toBe(1);
    expect(fetchCalls.some((call) => new globalThis.URL(call.url).pathname.endsWith("/checkpoint"))).toBe(true);
    // Recovering ChatGPT must retain the missing-page obligation for another
    // provider. Its later appearance needs a fresh exact-scope discovery.
    expect(fetchCalls.some((call) => new globalThis.URL(call.url).pathname.endsWith("/discover") &&
      captureJobRequestBody(call.options).provider === "claude-ai")).toBe(false);
    tabs = [...existingTabs, { id: 52, url: "https://claude.ai/new", title: "Claude" }];
    await sendRuntimeMessage({ type: "polylogue.backfill.status" });
    expect(fetchCalls.filter((call) => new globalThis.URL(call.url).pathname.endsWith("/discover") &&
      captureJobRequestBody(call.options).provider === "claude-ai")).toHaveLength(1);
    expect(providerWorkCalls).toBe(0);
  });

  it("keeps the newest same-provider recovery revision after a partial receiver commit", async () => {
    const checkpoint = (contentHash) => ({
      version: 1,
      jobs: [{
        id: "local-chatgpt", provider: "chatgpt", cutoff: "2026-01-01T00:00:00Z",
        status: "complete", inventory_cursor: "done", inventory_complete: true,
        policy: { leaseMs: 180000, maxDailyRequests: 10 }, execution_generation: 2,
        learned_cadence_ms: 40000, daily_requests: 3,
        cooldown_reason: null, last_error: null,
        last_ack: { receiver_request_id: `ack-${contentHash}`, content_hash: contentHash },
      }],
      queue: [{
        id: "queue-chatgpt", job_id: "local-chatgpt", provider: "chatgpt", native_id: "native-chatgpt",
        state: "complete", content_hash: contentHash, completed_at: "2026-07-16T10:00:00Z",
      }],
      revisions: [{
        id: "chatgpt:native-chatgpt", provider: "chatgpt", native_id: "native-chatgpt",
        provider_updated_at: "2026-07-16T09:00:00Z", content_hash: contentHash,
      }],
    });
    const receiverJobs = [
      {
        job_id: "receiver-new", provider: "chatgpt", intent_key: "intent-new", revision: 8,
        lease_generation: 2, checkpoint_sequence: 2, updated_at: "2026-07-16T10:06:00Z",
        checkpoint_updated_at: "2026-07-16T10:05:00Z",
        checkpoint: checkpointFixture(checkpoint("hash-new"), 2),
      },
      {
        job_id: "receiver-old", provider: "chatgpt", intent_key: "intent-old", revision: 4,
        lease_generation: 1, checkpoint_sequence: 1, updated_at: "2026-07-16T10:20:00Z",
        checkpoint_updated_at: "2026-07-16T10:00:00Z",
        checkpoint: checkpointFixture(checkpoint("hash-old"), 1),
      },
    ];
    let providerWorkCalls = 0;
    globalThis.chrome.scripting.executeScript = mockPageScript(async (details) => {
      const request = details.args?.[0];
      if (request?.operation === "identity") {
        return [{ result: { ok: true, response: { accountHandle: `${request.provider}-account` } } }];
      }
      providerWorkCalls += 1;
      return [{ result: { ok: false, error: "provider_work_must_not_replay" } }];
    });
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      fetchCalls.push({ url, options });
      const artifactResponse = checkpointArtifactFixtureResponse(url);
      if (artifactResponse) return artifactResponse;
      const path = new globalThis.URL(url).pathname;
      const body = captureJobRequestBody(options);
      if (path === "/v1/capture-jobs/capabilities") {
        return responseJson({
          schema: "polylogue.capture-jobs.capabilities.v1", checkpoint_transport: "canonical-artifact-v1", protocol_min: 2, protocol_max: 2,
          scope_namespace: "cjs1:revision-recovery-namespace",
        });
      }
      if (path === "/v1/capture-jobs/discover") {
        if (body.provider !== "chatgpt") return responseJson({ jobs: [] });
        return responseJson({ jobs: body.intent_key ? [receiverJobs[0]] : receiverJobs });
      }
      if (path.endsWith("/adopt")) {
        const recovered = receiverJobs.find((job) => path.includes(job.job_id));
        return responseJson({
          job: {
            ...recovered, revision: recovered.revision + 1,
            lease_generation: recovered.lease_generation + 1,
            updated_at: recovered.job_id === "receiver-old"
              ? "2026-07-16T10:10:00Z"
              : "2026-07-16T10:09:00Z",
          },
          lease: { lease_id: `lease-${recovered.job_id}`, generation: recovered.lease_generation + 1, proof: `proof-${recovered.job_id}` },
        });
      }
      if (path.endsWith("/update")) {
        return responseJson({
          job: {
            job_id: "receiver-new", provider: "chatgpt", intent_key: "intent-new", revision: 10,
            lease_generation: 3, lease_expires_at: "2099-01-01T00:00:00Z", checkpoint_sequence: 2,
          },
          receipt: { kind: "capture_job_update", revision: 10 },
        });
      }
      if (path.endsWith("/checkpoint")) {
        return checkpointFixtureResponse({ job_id: "receiver-new", revision: 11 }, options);
      }
      return responseJson({ error: "unexpected_receiver_request" }, { ok: false, status: 500 });
    });

    const status = await sendRuntimeMessage({ type: "polylogue.backfill.status" });

    expect(status.jobs[0].recovery_checkpoint_error).toBeNull();
    expect(status.jobs[0].last_ack.content_hash).toBe("hash-new");
    const committed = await Promise.all(fetchCalls
      .filter((call) => {
        const path = new globalThis.URL(call.url).pathname;
        return path.startsWith("/v1/capture-jobs/") && path.endsWith("/checkpoint");
      })
      .map(async (call) => JSON.parse(await call.options.body.text())));
    const latestCommitted = committed.at(-1);
    expect(latestCommitted.revisions).toEqual([expect.objectContaining({ content_hash: "hash-new" })]);
    expect(providerWorkCalls).toBe(0);
  });

  it("adopts and merges every provider checkpoint without replaying acknowledged work", async () => {
    tabs = [
      { id: 42, url: "https://chatgpt.com/?temporary-chat=true", title: "ChatGPT" },
      { id: 43, url: "https://claude.ai/new", title: "Claude" },
    ];
    const checkpoints = Object.fromEntries(["chatgpt", "claude-ai"].map((provider) => [provider, {
      version: 1,
      jobs: [{
        id: `local-${provider}`, provider, cutoff: "2026-01-01T00:00:00Z", status: "running",
        inventory_cursor: "done", inventory_complete: true,
        policy: { leaseMs: 180000, maxDailyRequests: 10, maxCapturesPerWake: 1 },
        execution_generation: 2, learned_cadence_ms: 40000, daily_requests: 3,
        last_ack: { receiver_request_id: `ack-${provider}`, content_hash: `hash-${provider}` },
      }],
      queue: [{
        id: `queue-${provider}`, job_id: `local-${provider}`, provider, native_id: `native-${provider}`,
        state: "complete", content_hash: `hash-${provider}`, completed_at: "2026-07-16T10:00:00Z",
      }],
      revisions: [{
        id: `${provider}:native-${provider}`, provider, native_id: `native-${provider}`,
        provider_updated_at: "2026-07-16T09:00:00Z", content_hash: `hash-${provider}`,
      }],
    }]));
    let providerWorkCalls = 0;
    globalThis.chrome.scripting.executeScript = mockPageScript(async (details) => {
      const request = details.args?.[0];
      if (request?.operation === "identity") {
        return [{ result: { ok: true, response: { accountHandle: `${request.provider}-account` } } }];
      }
      providerWorkCalls += 1;
      return [{ result: { ok: false, error: "provider_work_must_not_replay" } }];
    });
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      fetchCalls.push({ url, options });
      const artifactResponse = checkpointArtifactFixtureResponse(url);
      if (artifactResponse) return artifactResponse;
      const path = new globalThis.URL(url).pathname;
      const body = captureJobRequestBody(options);
      if (path === "/v1/capture-jobs/capabilities") {
        return responseJson({
          schema: "polylogue.capture-jobs.capabilities.v1", checkpoint_transport: "canonical-artifact-v1",
          protocol_min: 2,
          protocol_max: 2,
          scope_namespace: "cjs1:multi-provider-namespace",
        });
      }
      if (path === "/v1/browser-captures/capabilities") {
        return responseJson({ durable_ack_fields: ["receiver_request_id", "content_hash", "submitted_content_hash", "outcome"] });
      }
      if (path === "/v1/capture-jobs/discover") {
        const provider = body.provider;
        return responseJson({ jobs: checkpoints[provider] ? [{
          job_id: `receiver-${provider}`, provider, intent_key: `intent-${provider}`,
          revision: 4, lease_generation: 1, checkpoint_sequence: 1,
          updated_at: "2026-07-16T10:00:00Z", checkpoint: checkpointFixture(checkpoints[provider]),
        }] : [] });
      }
      if (path.endsWith("/adopt")) {
        const provider = body.provider;
        return responseJson({
          job: {
            job_id: `receiver-${provider}`, provider, intent_key: `intent-${provider}`,
            revision: 5, lease_generation: 2, checkpoint_sequence: 1,
            updated_at: "2026-07-16T10:00:00Z",
            checkpoint: checkpointFixture(checkpoints[provider]),
          },
          lease: { lease_id: `lease-${provider}`, generation: 2, proof: `proof-${provider}` },
        });
      }
      if (path.endsWith("/update")) {
        return responseJson({
          job: {
            job_id: path.split("/")[3], provider: body.provider, intent_key: `intent-${body.provider}`,
            revision: 6, lease_generation: 2, lease_expires_at: "2099-01-01T00:00:00Z",
            checkpoint_sequence: 1,
          },
          receipt: { kind: "capture_job_update", revision: 6 },
        });
      }
      if (path.endsWith("/checkpoint")) {
        return checkpointFixtureResponse({ job_id: path.split("/")[3], revision: 7 }, options);
      }
      return responseJson({ error: "unexpected_receiver_request" }, { ok: false, status: 500 });
    });

    const status = await sendRuntimeMessage({ type: "polylogue.backfill.status" });

    expect(status.jobs.map((job) => job.id).sort()).toEqual(["local-chatgpt", "local-claude-ai"]);
    expect(status.jobs.every((job) => job.status === "paused")).toBe(true);
    expect(status.jobs.map((job) => job.last_ack.receiver_request_id).sort()).toEqual([
      "ack-chatgpt", "ack-claude-ai",
    ]);
    const adoptedPaths = fetchCalls
      .filter((call) => new globalThis.URL(call.url).pathname.endsWith("/adopt"))
      .map((call) => new globalThis.URL(call.url).pathname);
    expect(new Set(adoptedPaths)).toEqual(new Set([
      "/v1/capture-jobs/receiver-chatgpt/adopt",
      "/v1/capture-jobs/receiver-claude-ai/adopt",
    ]));
    await sendRuntimeMessage({
      type: "polylogue.backfill.control", job_id: "local-chatgpt", action: "resume",
    });
    alarmListener({ name: "polylogueBackfillWake:local-chatgpt" });
    await vi.waitFor(async () => {
      const refreshed = await sendRuntimeMessage({ type: "polylogue.backfill.status" });
      expect(refreshed.jobs.find((job) => job.id === "local-chatgpt")?.status).toBe("complete");
    });
    expect(providerWorkCalls).toBe(0);
  });

  it("preserves a failed provider partition while reconciling another provider", async () => {
    const declaredClaudeScope = await deriveAccountScope("cjs1:partial-provider-namespace", "claude-ai", "declared-claude-account");
    const localJob = (id, provider) => ({
      id, provider, cutoff: "2026-01-01T00:00:00Z", status: "paused",
      inventory_cursor: "1", inventory_complete: false,
      policy: { leaseMs: 180000, maxDailyRequests: 10 }, execution_generation: 0,
      learned_cadence_ms: 40000, daily_requests: 1,
      cooldown_reason: "manual_pause", last_error: null, last_ack: null,
    });
    await loadBackground({
      polylogueBackfillRecoveryCheckpoint: {
        version: 1,
        jobs: [localJob("local-chatgpt", "chatgpt"), { ...localJob("local-claude", "claude-ai"), account_scope: declaredClaudeScope }],
        queue: [],
        revisions: [],
      },
    });
    tabs = [
      { id: 42, url: "https://chatgpt.com/", title: "ChatGPT" },
      { id: 43, url: "https://claude.ai/new", title: "Claude" },
    ];
    globalThis.chrome.scripting.executeScript = mockPageScript(async (details) => {
      const request = details.args?.[0];
      if (request?.operation === "identity" && request.provider === "chatgpt") {
        return [{ result: { ok: true, response: { accountHandle: "chatgpt-account" } } }];
      }
      if (request?.operation === "identity") {
        return [{ result: { ok: false, error: "claude_auth_unavailable" } }];
      }
      return [{ result: { ok: false, error: "provider_work_must_not_run" } }];
    });
    const remoteCheckpoint = {
      version: 1,
      jobs: [{
        ...localJob("remote-chatgpt", "chatgpt"), status: "complete",
        cooldown_reason: null, inventory_complete: true, last_ack: { receiver_request_id: "remote-ack" },
      }],
      queue: [],
      revisions: [],
    };
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      fetchCalls.push({ url, options });
      const artifactResponse = checkpointArtifactFixtureResponse(url);
      if (artifactResponse) return artifactResponse;
      const path = new globalThis.URL(url).pathname;
      const body = captureJobRequestBody(options);
      if (path === "/v1/capture-jobs/capabilities") {
        return responseJson({
          schema: "polylogue.capture-jobs.capabilities.v1", checkpoint_transport: "canonical-artifact-v1", protocol_min: 2, protocol_max: 2,
          scope_namespace: "cjs1:partial-provider-namespace",
        });
      }
      if (path === "/v1/capture-jobs/discover") {
        return responseJson({ jobs: body.provider === "chatgpt" ? [{
          job_id: "receiver-chatgpt", provider: "chatgpt", intent_key: "intent-chatgpt",
          revision: 4, lease_generation: 1, checkpoint_sequence: 1,
          updated_at: "2026-07-16T10:00:00Z", checkpoint_updated_at: "2026-07-16T09:59:00Z",
          checkpoint: checkpointFixture(remoteCheckpoint),
        }] : [] });
      }
      if (path.endsWith("/adopt")) {
        return responseJson({
          job: {
            job_id: "receiver-chatgpt", provider: "chatgpt", intent_key: "intent-chatgpt",
            revision: 5, lease_generation: 2, checkpoint_sequence: 1,
            updated_at: "2026-07-16T10:01:00Z", checkpoint: checkpointFixture(remoteCheckpoint),
          },
          lease: { lease_id: "lease-chatgpt", generation: 2, proof: "proof-chatgpt" },
        });
      }
      if (path.endsWith("/update")) {
        return responseJson({
          job: {
            job_id: "receiver-chatgpt", provider: "chatgpt", revision: 6, lease_generation: 2,
            lease_expires_at: "2099-01-01T00:00:00Z", checkpoint_sequence: 1,
          },
          receipt: {},
        });
      }
      if (path.endsWith("/checkpoint")) return checkpointFixtureResponse({ job_id: "receiver-chatgpt", revision: 7 }, options);
      return responseJson({ error: "unexpected_receiver_request" }, { ok: false, status: 500 });
    });

    const status = await sendRuntimeMessage({ type: "polylogue.backfill.status" });

    expect(status.jobs.map((job) => job.id).sort()).toEqual(["local-chatgpt", "local-claude", "remote-chatgpt"]);
    expect(status.jobs.find((job) => job.id === "remote-chatgpt")).toMatchObject({ status: "complete" });
    // Current account identity cannot reassign the unscoped historical job.
    expect(status.jobs.find((job) => job.id === "local-chatgpt")).toMatchObject({ status: "paused" });
    expect(status.jobs.find((job) => job.id === "local-claude")).toMatchObject({
      status: "paused", recovery_checkpoint_error: "claude_auth_unavailable",
    });
  });

  it("preserves local recovery without querying accountless legacy evidence", async () => {
    await loadBackground({
      polylogueBackfillRecoveryCheckpoint: {
        version: 1,
        jobs: [{
          id: "local-job", provider: "chatgpt", cutoff: "2026-01-01T00:00:00Z", status: "running",
          inventory_cursor: "3", policy: { leaseMs: 180000, maxDailyRequests: 10 }, execution_generation: 0,
          learned_cadence_ms: 40000, daily_requests: 1, last_ack: null,
        }],
        queue: [],
        revisions: [],
      },
    });
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      fetchCalls.push({ url, options });
      return responseJson({ error: "should_not_be_called" }, { ok: false, status: 500 });
    });

    await sendRuntimeMessage({ type: "polylogue.backfill.status" });

    expect(fetchCalls.some((call) => String(call.url).includes("/v1/backfill-checkpoint") && (call.options.method || "GET") === "GET")).toBe(false);
  });

  it("requires an existing provider surface before admitting a new account-scoped backfill", async () => {
    tabs = [{ id: 43, url: "https://help.chatgpt.com/article", title: "Help" }];

    const started = await sendRuntimeMessage({
      type: "polylogue.backfill.start",
      provider: "chatgpt",
      cutoff: "2026-01-01T00:00:00Z",
    });

    expect(started).toMatchObject({ ok: false, error: "provider_transport_no_surface" });
    expect((await sendRuntimeMessage({ type: "polylogue.backfill.status" })).jobs).toEqual([]);
    expect(globalThis.chrome.tabs.create).not.toHaveBeenCalled();
    expect(globalThis.chrome.scripting.executeScript).not.toHaveBeenCalledWith(expect.objectContaining({
      target: { tabId: expect.any(Number) },
      world: "MAIN",
    }));
  });

  it("preserves page-bridge Retry-After through the adapter and coordinator", async () => {
    globalThis.chrome.scripting.executeScript = mockPageScript(async (details) => {
      if (details.func) {
        const request = details.args?.[0];
        if (request?.operation === "identity") {
          return [{ result: { ok: true, response: { accountHandle: "retry-account" } } }];
        }
        return [{ result: { ok: true, response: {
          ok: false,
          status: 429,
          contentType: "application/json",
          retryAfter: "60",
          body: JSON.stringify({ detail: "slow down" }),
        } } }];
      }
      return undefined;
    });

    await sendRuntimeMessage({
      type: "polylogue.backfill.start",
      provider: "chatgpt",
      cutoff: "2026-01-01T00:00:00Z",
    });
    let status;
    await vi.waitFor(async () => {
      status = (await sendRuntimeMessage({ type: "polylogue.backfill.status" })).jobs[0];
      expect(status.cooldown_reason).toBe("provider_rate_limited");
    });

    const remainingCooldownMs = status.cooldown_until_ms - Date.parse(status.updated_at);
    expect(remainingCooldownMs).toBeGreaterThanOrEqual(59000);
    expect(remainingCooldownMs).toBeLessThanOrEqual(60000);
    expect(status.inventory_complete).toBe(false);
  });

  it("keeps an actual browser transport rejection retryable without an invented size hold", async () => {
    globalThis.chrome.scripting.executeScript = mockPageScript(async (details) => {
      if (details.args?.[0]?.operation === "identity") {
        return [{ result: { ok: true, response: { accountHandle: "oversize-account" } } }];
      }
      throw new Error("The message length exceeded the maximum allowed size");
    });

    const started = await sendRuntimeMessage({
      type: "polylogue.backfill.start",
      provider: "chatgpt",
      cutoff: "2026-01-01T00:00:00Z",
    });

    expect(started.ok).toBe(true);
    let status;
    await vi.waitFor(async () => {
      status = (await sendRuntimeMessage({ type: "polylogue.backfill.status" })).jobs[0];
      expect(status).toMatchObject({ status: "running", cooldown_reason: "transport_backoff" });
    });
    expect(status.last_error).toContain("The message length exceeded the maximum allowed size");
    // The failed page invocation was a real ChatGPT inventory request (cost
    // two); it is accounted once and enters retry backoff.
    expect(status.daily_requests).toBe(2);
    expect(status.transport_failures).toBe(1);
  });

  it("never creates a transport tab when passive backfill has no provider page", async () => {
    tabs = [];
    const started = await sendRuntimeMessage({
      type: "polylogue.backfill.start",
      provider: "chatgpt",
      cutoff: "2026-01-01T00:00:00Z",
    });
    expect(started).toMatchObject({ ok: false, error: "provider_transport_no_surface" });
    expect((await sendRuntimeMessage({ type: "polylogue.backfill.status" })).jobs).toEqual([]);
    await vi.waitFor(() => expect(globalThis.chrome.tabs.create).not.toHaveBeenCalled());
    expect(globalThis.chrome.tabs.remove).not.toHaveBeenCalled();
  });

  it("surfaces a stale Claude UI selection as a cancel-and-restart reason", async () => {
    tabs = [{ id: 52, url: "https://claude.ai/new", title: "Claude" }];
    globalThis.chrome.scripting.executeScript = mockPageScript(async (details) => {
      if (!details.func) return undefined;
      if (details.args?.[0]?.operation === "identity") {
        return [{ result: { ok: true, response: { accountHandle: "claude-account" } } }];
      }
      return [{ result: { ok: false, error: "backfill_bridge_selected_organization_stale" } }];
    });

    const started = await sendRuntimeMessage({
      type: "polylogue.backfill.start",
      provider: "claude-ai",
      cutoff: "2026-01-01T00:00:00Z",
    });
    expect(started.ok).toBe(true);
    let status;
    await vi.waitFor(async () => {
      status = (await sendRuntimeMessage({ type: "polylogue.backfill.status" })).jobs[0];
      expect(status.cooldown_reason).toBe("backfill_bridge_selected_organization_stale");
    });
    expect(status.inventory_complete).toBe(false);
  });

  it("captures an automatically detected missing conversation and records the decision timeline", async () => {
    expect(activatedListener).toBeTypeOf("function");
    tabs = [{ id: 42, url: "https://chatgpt.com/c/conv-123", title: "ChatGPT" }];
    stored.polylogueReceiverPairing = {
      state: "online",
      receiver_id: "rx-auto-capture",
      api_schema: "polylogue-browser-capture/v1",
      endpoint: "http://127.0.0.1:8875",
    };
    globalThis.fetch = vi.fn(async (url, options) => {
      fetchCalls.push({ url, options });
      if (String(url).endsWith("/v1/status")) {
        return responseJson({
          ok: true,
          receiver_id: "rx-auto-capture",
          api_schema: "polylogue-browser-capture/v1",
        });
      }
      if (String(url).endsWith("/v1/browser-captures")) {
        return captureReceipt({
          provider: "chatgpt",
          provider_session_id: "conv-123",
          state: "spooled_only",
          receiver_request_id: "capture-request-1",
        }, options, { requestId: "capture-request-1" });
      }
      return responseJson(
        {
          provider: "chatgpt",
          provider_session_id: "conv-123",
          state: "missing",
          lifecycle: "missing",
          captured: false,
          spooled: false,
          artifact_ref: "chatgpt/conv-123.json",
        },
        { requestId: "archive-state-1" },
      );
    });
    globalThis.chrome.tabs.sendMessage = vi.fn(async (tabId, message) => {
      if (message.type !== "polylogue.capturePage") return null;
      const envelope = {
        session: {
          provider: "chatgpt",
          provider_session_id: "conv-123",
          turns: [{ role: "user" }],
        },
      };
      const captureResult = await new Promise((resolve) => {
        messageListener(
          { type: "polylogue.capture", envelope, reason: message.reason },
          { tab: { id: tabId, url: "https://chatgpt.com/c/conv-123" } },
          resolve,
        );
      });
      return { ok: true, envelope, captureResult, archiveState: { state: "spooled_only" } };
    });

    activatedListener({ tabId: 42 });

    await vi.waitFor(() => expect(stored.polylogueState?.active_page_state).toBe("conversation"));
    await vi.waitFor(() => expect(stored.polylogueState?.captured).toBe(true));
    expect(fetchCalls.map((call) => call.url)).toContain(
      "http://127.0.0.1:8875/v1/archive-state?provider=chatgpt&provider_session_id=conv-123",
    );
    expect(stored.polylogueState.captured).toBe(true);
    expect(stored.polylogueState.last_receiver_request_id).toBe("capture-request-1");
    expect(globalThis.chrome.scripting.executeScript).toHaveBeenCalled();
    expect(globalThis.chrome.tabs.sendMessage).toHaveBeenCalledWith(42, {
      type: "polylogue.capturePage",
      reason: "auto_capture_missing",
    });
    expect(fetchCalls.map((call) => call.url)).toContain("http://127.0.0.1:8875/v1/browser-captures");
    const timeline = stored.polylogueConversationTimeline["chatgpt:conv-123"];
    expect(timeline.map((entry) => entry.event)).toEqual(["captured", "detected_new", "first_seen"]);
    expect(timeline[0]).toMatchObject({ reason: "auto_capture_missing", detail: "spooled_only" });
  });

  it("reconciles a temporary native identity across archive state and mission control without recapture", async () => {
    const url = "https://chatgpt.com/?temporary-chat=true";
    tabs = [{ id: 42, url }];
    stored.polylogueReceiverPairing = { state: "online", receiver_id: "rx-temporary", api_schema: "polylogue-browser-capture/v1", endpoint: "http://127.0.0.1:8875" };
    globalThis.chrome.tabs.sendMessage = vi.fn(async (_id, message) => message.type === "polylogue.captureIdentity" ? { provider_session_id: "temp-1" } : null);
    globalThis.fetch = vi.fn(async (input) => {
      fetchCalls.push({ url: String(input) });
      if (String(input).endsWith("/v1/status")) return responseJson({ ok: true, receiver_id: "rx-temporary", api_schema: "polylogue-browser-capture/v1" });
      if (new globalThis.URL(input).pathname === "/v1/archive-state") return responseJson({ provider: "chatgpt", provider_session_id: "temp-1", state: "archived", captured: true });
      return captureJobFixtureResponse(input) || responseJson({ ok: true });
    });
    activatedListener({ tabId: 42 });
    await vi.waitFor(() => expect(stored.polylogueState).toMatchObject({ provider_session_id: "temp-1", captured: true }));
    const snapshot = await sendRuntimeMessage({ type: "polylogue.missionControl.status", refresh: false }, { tab: tabs[0] });
    expect(snapshot.state).toMatchObject({ provider_session_id: "temp-1", captured: true, archive_state: { state: "archived" } });
    expect(snapshot.timeline.some((entry) => entry.detail === "already_safe")).toBe(true);
    expect(globalThis.chrome.tabs.sendMessage).toHaveBeenCalledWith(42, { type: "polylogue.captureIdentity", expectedUrl: url });
    expect(globalThis.chrome.tabs.sendMessage.mock.calls.some(([, message]) => message.type === "polylogue.capturePage")).toBe(false);
    const queries = fetchCalls.filter((call) => new globalThis.URL(call.url).pathname === "/v1/archive-state");
    expect(queries.length).toBeGreaterThan(0);
    expect(queries.every((call) => new globalThis.URL(call.url).searchParams.get("provider_session_id") === "temp-1")).toBe(true);
  });

  it.each([null, "bad/id", "__polylogue_temporary_chat__"])("keeps missing or invalid temporary identity unknown in mission control: %s", async (id) => {
    stored.polylogueState = { provider: "chatgpt", provider_session_id: null, captured: true };
    globalThis.chrome.tabs.sendMessage = vi.fn(async () => ({ provider_session_id: id }));
    const snapshot = await sendRuntimeMessage({ type: "polylogue.missionControl.status", refresh: false }, { tab: tabs[0] });
    expect(snapshot.state).toMatchObject({ provider_session_id: null, captured: false });
    expect(fetchCalls.some((call) => new globalThis.URL(call.url).pathname === "/v1/archive-state")).toBe(false);
  });

  it.each(["temp-1", "temp-2"])("rechecks temporary document identity before capturing a missing archive: %s", async (currentId) => {
    stored.polylogueReceiverPairing = { state: "online", receiver_id: "rx-temporary", api_schema: "polylogue-browser-capture/v1", endpoint: "http://127.0.0.1:8875" };
    let identityReads = 0;
    globalThis.chrome.tabs.sendMessage = vi.fn(async (_id, message) => {
      if (message.type === "polylogue.captureIdentity") return { provider_session_id: ++identityReads === 1 ? "temp-1" : currentId };
      return { ok: false, error: "fixture_capture_stop" };
    });
    globalThis.fetch = vi.fn(async (input) => {
      fetchCalls.push({ url: String(input) });
      if (String(input).endsWith("/v1/status")) return responseJson({ ok: true, receiver_id: "rx-temporary", api_schema: "polylogue-browser-capture/v1" });
      if (new globalThis.URL(input).pathname === "/v1/archive-state") return responseJson({ provider: "chatgpt", provider_session_id: "temp-1", state: "missing", captured: false });
      return captureJobFixtureResponse(input) || responseJson({ ok: true });
    });
    await sendRuntimeMessage({ type: "polylogue.missionControl.status" }, { tab: tabs[0] });
    expect(identityReads).toBeGreaterThanOrEqual(2);
    const captures = globalThis.chrome.tabs.sendMessage.mock.calls.filter(([, message]) => message.type === "polylogue.capturePage");
    expect(captures).toHaveLength(currentId === "temp-1" ? 1 : 0);
    if (currentId !== "temp-1") expect(stored.polylogueConversationTimeline["chatgpt:temp-1"]).toContainEqual(expect.objectContaining({ event: "held_with_reason", detail: "tab_navigation_changed" }));
  });

  it("reports temporary-chat identity transport failure through the archived-tab state owner", async () => {
    tabs = [{ id: 42, url: "https://chatgpt.com/?temporary-chat=true", title: "ChatGPT", active: true }];
    globalThis.chrome.tabs.sendMessage = vi.fn(async (_tabId, message) => {
      if (message.type === "polylogue.captureIdentity") throw new Error("synthetic_identity_transport_failure");
      return { ok: true };
    });
    activatedListener({ tabId: 42 });
    await vi.waitFor(() => expect(stored.polylogueState).toMatchObject({
      error: "synthetic_identity_transport_failure", provider: "chatgpt", provider_session_id: null,
    }));
    expect(fetchCalls.some((call) => String(call.url).includes("/v1/archive-state"))).toBe(false);
    expect(globalThis.chrome.tabs.sendMessage.mock.calls.some(([, message]) => message.type === "polylogue.capturePage")).toBe(false);
  });

  it("captures a ChatGPT temporary chat instead of silently skipping it (background.js conversationIdForUrl asymmetry)", async () => {
    // background.js's own conversationIdForUrl used to return null for a
    // temporary-chat URL (?temporary-chat=true), which made captureTab's
    // automatic-capture gate (`!conversationIdForUrl(conversationUrl)`)
    // treat every temporary chat tab as having no session and silently
    // never send it a polylogue.capturePage message at all -- content
    // script capture code was ready for temporary chats the whole time.
    // Zero temporary chats had ever landed in the archive because of this.
    expect(activatedListener).toBeTypeOf("function");
    tabs = [{ id: 42, url: "https://chatgpt.com/?temporary-chat=true", title: "ChatGPT" }];
    stored.polylogueReceiverPairing = {
      state: "online",
      receiver_id: "rx-auto-capture",
      api_schema: "polylogue-browser-capture/v1",
      endpoint: "http://127.0.0.1:8875",
    };
    globalThis.fetch = vi.fn(async (url) => {
      fetchCalls.push({ url });
      if (String(url).endsWith("/v1/status")) {
        return responseJson({ ok: true, receiver_id: "rx-auto-capture", api_schema: "polylogue-browser-capture/v1" });
      }
      if (String(url).endsWith("/v1/browser-captures")) {
        return responseJson({
          provider: "chatgpt",
          provider_session_id: "temp:ephemeral-session",
          state: "spooled_only",
          receiver_request_id: "capture-request-temp",
        }, { requestId: "capture-request-temp" });
      }
      return responseJson(
        { provider: "chatgpt", provider_session_id: "temp:ephemeral-session", state: "missing", lifecycle: "missing", captured: false, spooled: false },
        { requestId: "archive-state-temp" },
      );
    });
    globalThis.chrome.tabs.sendMessage = vi.fn(async (tabId, message) => {
      if (message.type !== "polylogue.capturePage") return null;
      const envelope = {
        session: { provider: "chatgpt", provider_session_id: "temp:ephemeral-session", session_kind: "temporary", turns: [{ role: "user" }] },
      };
      const captureResult = await new Promise((resolve) => {
        messageListener(
          { type: "polylogue.capture", envelope, reason: message.reason },
          { tab: { id: tabId, url: "https://chatgpt.com/?temporary-chat=true" } },
          resolve,
        );
      });
      return { ok: true, envelope, captureResult, archiveState: { state: "spooled_only" } };
    });

    activatedListener({ tabId: 42 });

    await vi.waitFor(() => expect(globalThis.chrome.tabs.sendMessage).toHaveBeenCalled());
    expect(globalThis.chrome.tabs.sendMessage).toHaveBeenCalledWith(42, expect.objectContaining({ type: "polylogue.capturePage" }));
    await vi.waitFor(() => expect(stored.polylogueState?.captured).toBe(true));
    expect(fetchCalls.map((call) => call.url)).toContain("http://127.0.0.1:8875/v1/browser-captures");
  });

  it("queries a temporary chat using its current content-script identity", async () => {
    tabs = [{ id: 42, url: "https://chatgpt.com/?temporary-chat=true" }];
    globalThis.chrome.tabs.sendMessage = vi.fn(async (_id, message) => {
      if (message.type === "polylogue.captureIdentity") return { provider_session_id: "temp-1" };
      throw new Error("archived_temporary_chat_must_not_recapture");
    });
    globalThis.fetch = vi.fn(async (url) => {
      fetchCalls.push({ url });
      return responseJson({ state: "archived", captured: true, provider: "chatgpt", provider_session_id: "temp-1" });
    });
    activatedListener({ tabId: 42 });
    await vi.waitFor(() => expect(stored.polylogueState?.provider_session_id).toBe("temp-1"));
    const archiveRequest = fetchCalls.find((call) => new globalThis.URL(call.url).pathname === "/v1/archive-state");
    expect(new globalThis.URL(archiveRequest.url).searchParams.get("provider_session_id")).toBe("temp-1");
    expect(globalThis.chrome.tabs.sendMessage).not.toHaveBeenCalledWith(42, expect.objectContaining({ type: "polylogue.capturePage" }));
  });

  it("does not enumerate authenticated inventory during an unpaired freshness sweep", async () => {
    stored.polylogueReceiverPairing = null;
    alarmListener({ name: "polylogueCaptureFreshnessSweep" });
    await vi.waitFor(() => expect(globalThis.chrome.storage.local.get).toHaveBeenCalled());
    await new Promise((resolve) => globalThis.setTimeout(resolve, 25));
    expect(globalThis.chrome.scripting.executeScript).not.toHaveBeenCalled();
  });

  it("does not recapture an already-safe conversation on activation", async () => {
    tabs = [{ id: 42, url: "https://chatgpt.com/c/conv-throttle", title: "ChatGPT" }];
    globalThis.fetch = vi.fn(async () => responseJson({
      provider: "chatgpt",
      provider_session_id: "conv-throttle",
      state: "archived",
      captured: true,
    }));

    activatedListener({ tabId: 42 });
    await vi.waitFor(() => expect(stored.polylogueState?.archive_state?.state).toBe("archived"));
    activatedListener({ tabId: 42 });
    await new Promise((resolve) => globalThis.setTimeout(resolve, 20));

    expect(globalThis.chrome.tabs.sendMessage).not.toHaveBeenCalled();
    expect((stored.polylogueConversationTimeline?.["chatgpt:conv-throttle"] || [])
      .some((event) => event.detail === "background_capture_throttled")).toBe(false);
  });

  it.each(["archived", "spooled_only"])("installs observers on an existing %s ChatGPT tab without recapturing", async (state) => {
    const pairing = { state: "online", receiver_id: "rx-installed-observer",
      api_schema: "polylogue-browser-capture/v1", endpoint: "http://127.0.0.1:8875" };
    await loadBackground({ polylogueReceiverPairing: pairing });
    tabs = [{ id: 42, url: "https://chatgpt.com/c/installed-observer", title: "ChatGPT" }];
    globalThis.fetch = vi.fn(async (url) => {
      fetchCalls.push({ url });
      if (String(url).endsWith("/v1/status")) {
        return responseJson({ ok: true, receiver_id: pairing.receiver_id, api_schema: pairing.api_schema });
      }
      return responseJson({ provider: "chatgpt", provider_session_id: "installed-observer",
        state, captured: state === "archived", spooled: true });
    });

    installedListener();
    await vi.waitFor(() => expect(stored.polylogueState?.archive_state?.state).toBe(state));
    expect(globalThis.chrome.scripting.executeScript).toHaveBeenCalledWith({
      target: { tabId: 42 }, files: ["src/content/asset_stream.js", "src/content/chatgpt_bridge.js"], world: "MAIN",
    });
    expect(globalThis.chrome.scripting.executeScript.mock.calls.some(([details]) =>
      details.target.tabId === 42 && details.files?.includes("src/content/chatgpt.js"))).toBe(true);
    expect(globalThis.chrome.tabs.sendMessage).not.toHaveBeenCalledWith(42,
      expect.objectContaining({ type: "polylogue.capturePage" }));
    expect(fetchCalls.every(({ url }) => new globalThis.URL(String(url)).hostname === "127.0.0.1")).toBe(true);
  });

  it("never fetches a terminal ChatGPT conversation across a receiver outage and a service-worker restart", async () => {
    const pairing = {
      state: "online",
      receiver_id: "rx-restart-guard",
      api_schema: "polylogue-browser-capture/v1",
      endpoint: "http://127.0.0.1:8875",
    };
    await loadBackground({ polylogueReceiverPairing: pairing });
    tabs = [{ id: 42, url: "https://chatgpt.com/c/conv-terminal-restart", title: "ChatGPT" }];
    globalThis.fetch = vi.fn(async (url) => {
      if (String(url).endsWith("/v1/status")) {
        return responseJson({ ok: true, receiver_id: pairing.receiver_id, api_schema: pairing.api_schema });
      }
      return responseJson({
        provider: "chatgpt",
        provider_session_id: "conv-terminal-restart",
        state: "archived",
        captured: true,
      });
    });

    installedListener();
    await vi.waitFor(() => expect(stored.polylogueState?.archive_state?.state).toBe("archived"));
    expect(globalThis.chrome.tabs.sendMessage).not.toHaveBeenCalled();

    // Receiver goes down, then the extension's own service worker restarts.
    // chrome.storage.local (the pairing) survives; in-memory reconciliation
    // state does not — this is the shape of the original incident.
    await loadBackground({ polylogueReceiverPairing: pairing });
    tabs = [{ id: 42, url: "https://chatgpt.com/c/conv-terminal-restart", title: "ChatGPT" }];
    globalThis.fetch = vi.fn(async () => {
      throw new TypeError("Failed to fetch");
    });

    installedListener();
    await vi.waitFor(() => expect(stored.polylogueState?.online).toBe(false));

    expect(globalThis.chrome.tabs.sendMessage).not.toHaveBeenCalled();
    expect(globalThis.chrome.scripting.executeScript).toHaveBeenCalledWith(expect.objectContaining({ target: { tabId: 42 }, files: expect.arrayContaining(["src/common.js", "src/content/chatgpt.js"]) }));
    expect(globalThis.chrome.scripting.executeScript.mock.calls.every(([details]) => Array.isArray(details.files))).toBe(true);
  });

  it("recaptures an archived Claude conversation until that provider has freshness convergence", async () => {
    tabs = [{ id: 42, url: "https://claude.ai/chat/claude-freshness", title: "Claude" }];
    stored.polylogueReceiverPairing = {
      state: "online",
      receiver_id: "rx-claude-capture",
      api_schema: "polylogue-browser-capture/v1",
      endpoint: "http://127.0.0.1:8875",
    };
    globalThis.fetch = vi.fn(async (url) => {
      if (String(url).endsWith("/v1/status")) {
        return responseJson({
          ok: true,
          receiver_id: "rx-claude-capture",
          api_schema: "polylogue-browser-capture/v1",
        });
      }
      return responseJson({
        provider: "claude-ai",
        provider_session_id: "claude-freshness",
        state: "archived",
        captured: true,
      });
    });

    activatedListener({ tabId: 42 });

    await vi.waitFor(() => expect(globalThis.chrome.tabs.sendMessage).toHaveBeenCalledWith(42, {
      type: "polylogue.capturePage",
      reason: "auto_capture_unconverged_provider",
    }));
    expect(globalThis.chrome.tabs.sendMessage).toHaveBeenCalledTimes(1);
  });

  it("captures a missing conversation once during automatic reconciliation", async () => {
    tabs = [{ id: 42, url: "https://chatgpt.com/c/conv-missing-on-start", title: "ChatGPT" }];
    stored.polylogueReceiverPairing = {
      state: "online",
      receiver_id: "rx-auto-capture",
      api_schema: "polylogue-browser-capture/v1",
      endpoint: "http://127.0.0.1:8875",
    };
    globalThis.fetch = vi.fn(async (url) => {
      if (String(url).endsWith("/v1/status")) {
        return responseJson({
          ok: true,
          receiver_id: "rx-auto-capture",
          api_schema: "polylogue-browser-capture/v1",
        });
      }
      if (String(url).includes("/v1/archive-state")) {
        return responseJson({
          provider: "chatgpt",
          provider_session_id: "conv-missing-on-start",
          state: "missing",
          captured: false,
        });
      }
      return responseJson({ error: "unexpected" }, { ok: false, status: 500 });
    });

    installedListener();

    await vi.waitFor(() => expect(globalThis.chrome.tabs.sendMessage).toHaveBeenCalledWith(42, {
      type: "polylogue.capturePage",
      reason: "auto_capture_missing",
    }));
  });

  it("does not read a provider conversation from an unpaired freshness hint", async () => {
    vi.spyOn(Date, "now").mockReturnValue(100_000);
    await sendRuntimeMessage({
      type: "polylogue.captureFreshnessHint",
      provider: "chatgpt",
      provider_session_id: "unpaired-freshness",
      reason: "generation_completed",
      delay_ms: 0,
    });

    alarmListener({ name: "polylogueCaptureFreshnessWake" });

    await vi.waitFor(() => expect(
      stored.polylogueCaptureFreshnessQueue.entries["chatgpt:unpaired-freshness"]?.last_error,
    ).toBe("receiver_unpaired"));
    expect(globalThis.chrome.tabs.create).not.toHaveBeenCalled();
    expect(globalThis.chrome.tabs.sendMessage).not.toHaveBeenCalled();
  });

  it("accepts a temporary chat's freshness hint even though its real ephemeral id differs from the URL sentinel", async () => {
    // background.js's conversationIdForUrl returns TEMPORARY_CHAT_SENTINEL
    // for a ChatGPT temporary-chat URL (there is no /c/<id> to read), but
    // chatgpt.js's freshness hints always carry the conversation's true
    // ephemeral id (read from the intercepted native payload). Before this
    // fix, the sender-identity check compared the sentinel against that real
    // id, found a mismatch, and threw -- rejecting every freshness hint a
    // temporary chat ever sent after its first capture, so later turns were
    // never re-captured (P1 finding on PR #3411's Codex review).
    vi.spyOn(Date, "now").mockReturnValue(100_000);
    const response = await sendRuntimeMessage(
      {
        type: "polylogue.captureFreshnessHint",
        provider: "chatgpt",
        provider_session_id: "temp-conv-ephemeral-42",
        reason: "generation_completed",
        delay_ms: 0,
      },
      { tab: { id: 42, url: "https://chatgpt.com/?temporary-chat=true" } },
    );

    expect(response.ok).toBe(true);
    expect(stored.polylogueCaptureFreshnessQueue.entries["chatgpt:temp-conv-ephemeral-42"]).toBeDefined();
  });

  it("still rejects a freshness hint whose id does not match an ordinary (non-temporary) tab's own URL", async () => {
    vi.spyOn(Date, "now").mockReturnValue(100_000);
    const response = await sendRuntimeMessage(
      {
        type: "polylogue.captureFreshnessHint",
        provider: "chatgpt",
        provider_session_id: "some-other-conversation",
        reason: "generation_completed",
        delay_ms: 0,
      },
      { tab: { id: 42, url: "https://chatgpt.com/c/conv-actually-open" } },
    );

    expect(response).toMatchObject({ ok: false, error: "freshness_hint_sender_identity_mismatch" });
  });

  it("anti-vacuity: a Gemini freshness hint reaches the Gemini content script", async () => {
    // Without a Gemini branch in conversationIdForUrl, captureTab returns null
    // before messaging the tab and the hint was still reported as ok.
    const geminiUrl = "https://gemini.google.com/app/gemini-fresh";
    tabs = [{ id: 42, url: geminiUrl, title: "Gemini" }];
    stored.polylogueReceiverPairing = {
      state: "online",
      receiver_id: "rx-auto-capture",
      api_schema: "polylogue-browser-capture/v1",
      endpoint: "http://127.0.0.1:8875",
    };
    globalThis.fetch = vi.fn(async (url) => {
      if (String(url).endsWith("/v1/status")) {
        return responseJson({ ok: true, receiver_id: "rx-auto-capture", api_schema: "polylogue-browser-capture/v1" });
      }
      return responseJson(
        { provider: "gemini", provider_session_id: "gemini-fresh", state: "archived", lifecycle: "archived", captured: true, spooled: false },
        { requestId: "archive-state-gemini" },
      );
    });
    globalThis.chrome.tabs.sendMessage = vi.fn(async (_tabId, message) => (
      message.type === "polylogue.capturePage" ? { ok: true, archiveState: { state: "archived" } } : null
    ));

    await sendRuntimeMessage(
      {
        type: "polylogue.captureFreshnessHint",
        provider: "gemini",
        provider_session_id: "gemini-fresh",
        reason: "provider_turns_changed",
        delay_ms: 0,
      },
      { tab: { id: 42, url: geminiUrl } },
    );

    expect(globalThis.chrome.tabs.sendMessage).toHaveBeenCalledWith(
      42,
      expect.objectContaining({ type: "polylogue.capturePage" }),
    );
  });

  it("drains every supported provider capture owner when automatic capture is paused", async () => {
    tabs = [
      { id: 41, url: "https://chatgpt.com/c/paused" },
      { id: 42, url: "https://claude.ai/chat/paused" },
      { id: 43, url: "https://grok.com/c/paused" },
      { id: 44, url: "https://gemini.google.com/app/paused" },
      { id: 45, url: "https://example.com/" },
    ];
    expect(await sendRuntimeMessage({ type: "polylogue.ambient.configure", automatic_capture_enabled: false }))
      .toMatchObject({ ok: true, ambient: { automatic_capture_enabled: false } });
    expect(globalThis.chrome.tabs.sendMessage.mock.calls).toEqual([
      [41, { type: "polylogue.cancelCapture" }], [42, { type: "polylogue.cancelCapture" }],
      [43, { type: "polylogue.cancelCapture" }], [44, { type: "polylogue.cancelCapture" }],
    ]);
  });

  it("does not fetch a missing conversation while automatic capture is paused", async () => {
    tabs = [{ id: 42, url: "https://chatgpt.com/c/conv-paused", title: "ChatGPT" }];
    await sendRuntimeMessage({
      type: "polylogue.ambient.configure",
      automatic_capture_enabled: false,
    });
    globalThis.fetch = vi.fn(async () => responseJson({
      provider: "chatgpt",
      provider_session_id: "conv-paused",
      state: "missing",
      captured: false,
    }));

    installedListener();

    await vi.waitFor(() => expect(stored.polylogueSessionLedger["chatgpt:conv-paused"]?.archive_state)
      .toMatchObject({ state: "missing" }));
    expect(globalThis.chrome.tabs.sendMessage.mock.calls.every(([, message]) => message.type === "polylogue.cancelCapture")).toBe(true);
    expect(globalThis.chrome.alarms.clear).toHaveBeenCalledWith("polylogueCaptureFreshnessWake");
  });

  it("keeps the earliest freshness deadline when a later hint is queued", async () => {
    vi.spyOn(Date, "now").mockReturnValue(100_000);
    await sendRuntimeMessage({
      type: "polylogue.captureFreshnessHint",
      provider: "chatgpt",
      provider_session_id: "earliest",
      reason: "first",
      delay_ms: 1_000,
    });
    await sendRuntimeMessage({
      type: "polylogue.captureFreshnessHint",
      provider: "chatgpt",
      provider_session_id: "later",
      reason: "second",
      delay_ms: 60_000,
    });

    const freshnessAlarms = globalThis.chrome.alarms.create.mock.calls
      .filter(([name]) => name === "polylogueCaptureFreshnessWake");
    expect(freshnessAlarms.at(-1)[1]).toEqual({ when: 101_000 });
  });

  it("persists terminal lifecycle evidence with an immediate freshness deadline", async () => {
    vi.spyOn(Date, "now").mockReturnValue(100_000);
    const observation = {
      observation_id: "terminal:turn-2",
      state: "completed",
      observed_at: "2026-07-16T01:26:30Z",
      displayed_elapsed_ms: 5_190_000,
    };

    await sendRuntimeMessage({
      type: "polylogue.captureFreshnessHint",
      provider: "chatgpt",
      provider_session_id: "terminal-conversation",
      reason: "generation_completed",
      delay_ms: 0,
      generation_observations: [observation],
    });

    expect(stored.polylogueCaptureFreshnessQueue.entries["chatgpt:terminal-conversation"])
      .toMatchObject({
        next_attempt_at_ms: 100_000,
        generation_observations: [observation],
      });
    const freshnessAlarms = globalThis.chrome.alarms.create.mock.calls
      .filter(([name]) => name === "polylogueCaptureFreshnessWake");
    expect(freshnessAlarms.at(-1)[1]).toEqual({ when: 101_000 });
  });

  it("applies a typed freshness rate limit to every ChatGPT conversation", async () => {
    const now = vi.spyOn(Date, "now").mockReturnValue(100_000);
    tabs = [{ id: 42, url: "https://chatgpt.com/c/throttled-one", title: "ChatGPT" }];
    stored.polylogueReceiverPairing = {
      state: "online",
      receiver_id: "rx-freshness-throttle",
      api_schema: "polylogue-browser-capture/v1",
      endpoint: "http://127.0.0.1:8875",
    };
    globalThis.fetch = vi.fn(async (url) => {
      if (String(url).endsWith("/v1/status")) {
        return responseJson({ ok: true, receiver_id: "rx-freshness-throttle", api_schema: "polylogue-browser-capture/v1" });
      }
      throw new Error(`unexpected receiver request: ${url}`);
    });
    globalThis.chrome.tabs.sendMessage = vi.fn(async () => ({
      ok: false,
      error: "rate_limited",
      outcome: "rate_limited",
      retry_after_seconds: 73,
    }));

    await sendRuntimeMessage({
      type: "polylogue.captureFreshnessHint",
      provider: "chatgpt",
      provider_session_id: "throttled-one",
      reason: "generation_completed",
      delay_ms: 0,
    });
    await sendRuntimeMessage({
      type: "polylogue.captureFreshnessHint",
      provider: "chatgpt",
      provider_session_id: "throttled-two",
      reason: "generation_completed",
      delay_ms: 0,
    });

    alarmListener({ name: "polylogueCaptureFreshnessWake" });
    await vi.waitFor(() => expect(stored.polylogueCaptureFreshnessQueue.provider_cooldowns.chatgpt).toBe(173_000));
    expect(globalThis.chrome.tabs.sendMessage).toHaveBeenCalledTimes(1);

    now.mockReturnValue(101_000);
    alarmListener({ name: "polylogueCaptureFreshnessWake" });
    await new Promise((resolve) => globalThis.setTimeout(resolve, 20));
    expect(globalThis.chrome.tabs.sendMessage).toHaveBeenCalledTimes(1);
    const freshnessAlarms = globalThis.chrome.alarms.create.mock.calls
      .filter(([name]) => name === "polylogueCaptureFreshnessWake");
    expect(freshnessAlarms.at(-1)[1]).toEqual({ when: 173_000 });
  });

  // Anti-vacuity: restore `Number(value.retry_after_seconds) || null` and an
  // Infinity Retry-After passes through, so no finite cooldown is stored.
  it("uses the default rate-limit delay for a non-finite resolved Retry-After", async () => {
    vi.spyOn(Date, "now").mockReturnValue(100_000);
    tabs = [{ id: 42, url: "https://chatgpt.com/c/infinite-retry", title: "ChatGPT" }];
    stored.polylogueReceiverPairing = {
      state: "online",
      receiver_id: "rx-infinite-retry",
      api_schema: "polylogue-browser-capture/v1",
      endpoint: "http://127.0.0.1:8875",
    };
    globalThis.fetch = vi.fn(async (url) => {
      if (String(url).endsWith("/v1/status")) {
        return responseJson({ ok: true, receiver_id: "rx-infinite-retry", api_schema: "polylogue-browser-capture/v1" });
      }
      throw new Error(`unexpected receiver request: ${url}`);
    });
    globalThis.chrome.tabs.sendMessage = vi.fn(async () => ({
      ok: false,
      error: "rate_limited",
      outcome: "rate_limited",
      retry_after_seconds: Infinity,
    }));

    await sendRuntimeMessage({
      type: "polylogue.captureFreshnessHint",
      provider: "chatgpt",
      provider_session_id: "infinite-retry",
      reason: "generation_completed",
      delay_ms: 0,
    });

    alarmListener({ name: "polylogueCaptureFreshnessWake" });
    await vi.waitFor(() => expect(stored.polylogueCaptureFreshnessQueue.provider_cooldowns.chatgpt)
      .toBe(100_000 + 15 * 60_000));
  });

  it("refuses an unrepresentable provider deadline without storing Infinity", async () => {
    vi.spyOn(Date, "now").mockReturnValue(100_000);
    const result = await sendRuntimeMessage({ type: "polylogue.providerRateLimited", provider: "chatgpt", request_id: "actual-response", provider_response: { status: 429, url: "https://chatgpt.com/backend-api/conversation/synthetic" }, retry_after_seconds: 1e307 }, { tab: tabs[0], documentId: "synthetic" });
    expect(result).toMatchObject({ ok: false, error: "provider_retry_after_unrepresentable" });
    expect(stored.polylogueCaptureFreshnessQueue?.provider_cooldowns?.chatgpt).toBeUndefined();
  });

  it("persists a content-reported rate limit before another conversation can capture", async () => {
    vi.spyOn(Date, "now").mockReturnValue(100_000);
    tabs = [{ id: 42, url: "https://chatgpt.com/c/content-rate-limit", title: "ChatGPT" }];
    await sendRuntimeMessage({
      type: "polylogue.providerRateLimited",
      provider: "chatgpt", request_id: "actual-response",
      provider_response: { status: 429, url: "https://chatgpt.com/backend-api/conversation/synthetic" },
      retry_after_seconds: 73,
    }, { tab: tabs[0], documentId: "synthetic" });
    await sendRuntimeMessage({
      type: "polylogue.captureFreshnessHint",
      provider: "chatgpt",
      provider_session_id: "other-conversation",
      reason: "generation_completed",
      delay_ms: 0,
    });

    alarmListener({ name: "polylogueCaptureFreshnessWake" });
    await new Promise((resolve) => globalThis.setTimeout(resolve, 20));

    expect(stored.polylogueCaptureFreshnessQueue.provider_cooldowns.chatgpt).toBe(173_000);
    expect(globalThis.chrome.tabs.sendMessage).not.toHaveBeenCalled();
  });

  it("retains a forty-eight hour provider cooldown across conversations without requesting assets or fallback", async () => {
    vi.spyOn(Date, "now").mockReturnValue(100_000);
    const sender = { tab: tabs[0], documentId: "synthetic" };
    expect(await sendRuntimeMessage({ type: "polylogue.providerRateLimited", provider: "chatgpt", request_id: "actual-response", provider_response: { status: 429, url: "https://chatgpt.com/backend-api/conversation/synthetic" }, retry_after_seconds: 172800 }, sender)).toMatchObject({ ok: true });
    expect(stored.polylogueCaptureFreshnessQueue.provider_cooldowns.chatgpt).toBe(100_000 + 172800000);
    expect(await sendRuntimeMessage({ type: "polylogue.providerThrottle", provider: "chatgpt" })).toMatchObject({ ok: false, outcome: "rate_limited", retry_after_seconds: 172800 });
    expect(await sendRuntimeMessage({ type: "polylogue.providerRateLimited", provider: "claude-ai", request_id: "actual-response", provider_response: { status: 429, url: "https://claude.ai/api/organizations/synthetic" }, retry_after_seconds: 60 }, sender)).toMatchObject({ ok: false, error: "provider_rate_limit_sender_invalid" });
    expect(globalThis.chrome.tabs.sendMessage).not.toHaveBeenCalled();
  });

  it("persists the actual page identity429 before any backfill inventory or conversation request", async () => {
    vi.spyOn(Date, "now").mockReturnValue(100_000);
    tabs = [{ id: 42, url: "https://chatgpt.com/", title: "ChatGPT" }];
    const requests = [];
    globalThis.chrome.scripting.executeScript = mockPageScript(async (details) => {
      if (!details.args?.[0]) return [{ result: true }];
      requests.push(details.args[0]);
      return [{ result: { ok: false, error: "provider_rate_limited", outcome: "rate_limited", status: 429,
        retryAfter: "172800", responseUrl: "https://chatgpt.com/api/auth/session" } }];
    });
    const result = await sendRuntimeMessage({ type: "polylogue.backfill.start", provider: "chatgpt", cutoff: "2026-01-01T00:00:00Z" });
    expect(result).toMatchObject({ ok: false, outcome: "rate_limited" });
    expect(stored.polylogueCaptureFreshnessQueue.provider_cooldowns.chatgpt).toBe(100_000 + 172800000);
    expect(requests).toHaveLength(1);
    expect(requests[0].operation).toBe("identity");
    expect(globalThis.chrome.tabs.sendMessage).not.toHaveBeenCalled();
  });

  it.each(["native-response", "provider-inventory"])("records a staged %s rate limit against its admitted request", async (kind) => {
    vi.spyOn(Date, "now").mockReturnValue(100_000);
    const sender = { tab: tabs[0], documentId: "owned-document" };
    const owner = { tab_id: sender.tab.id, document_id: sender.documentId, provider: "chatgpt" };
    const staging = new CaptureStaging(globalThis.navigator.storage, new IndexedDbBackfillStore(globalThis.indexedDB));
    const sourceUrl = kind === "native-response" ? "https://chatgpt.com/backend-api/conversation/session" : "https://chatgpt.com/backend-api/conversations";
    const claim = await staging.begin(owner, { kind, source_url: sourceUrl }, "admitted-request");
    const report = { type: "polylogue.providerRateLimited", provider: "chatgpt", claim,
      request_id: "admitted-request", provider_response: { status: 429, url: sourceUrl }, retry_after: "172800" };
    expect(await sendRuntimeMessage({ ...report, request_id: "another-request" }, sender))
      .toMatchObject({ ok: false, error: "provider_rate_limit_claim_invalid" });
    expect(await sendRuntimeMessage({ ...report, provider_response: { status: 429, url: `${sourceUrl}/another` } }, sender))
      .toMatchObject({ ok: false, error: "provider_rate_limit_claim_invalid" });
    expect(stored.polylogueCaptureFreshnessQueue?.provider_cooldowns?.chatgpt).toBeUndefined();
    expect(await sendRuntimeMessage(report, sender)).toMatchObject({ ok: true });
    expect(stored.polylogueCaptureFreshnessQueue.provider_cooldowns.chatgpt).toBe(100_000 + 172800000);
    expect(await sendRuntimeMessage({ type: "polylogue.providerThrottle", provider: "chatgpt" }))
      .toMatchObject({ ok: false, outcome: "rate_limited", retry_after_seconds: 172800 });
    expect(globalThis.chrome.tabs.sendMessage).not.toHaveBeenCalled();
  });

  it("serializes concurrent captures without losing either ledger or timeline entry", async () => {
    globalThis.fetch = vi.fn(async (_url, options) => {
      const session = JSON.parse(await options.body.text()).session;
      return captureReceipt({ provider: session.provider, provider_session_id: session.provider_session_id }, options);
    });

    await Promise.all(["conv-a", "conv-b"].map((providerSessionId) => sendRuntimeMessage({
      type: "polylogue.capture",
      envelope: { session: { provider: "chatgpt", provider_session_id: providerSessionId } },
    })));

    expect(Object.keys(stored.polylogueSessionLedger).sort()).toEqual(["chatgpt:conv-a", "chatgpt:conv-b"]);
    expect(Object.keys(stored.polylogueConversationTimeline).sort()).toEqual(["chatgpt:conv-a", "chatgpt:conv-b"]);
  });

  it("records a held decision when automatic capture is throttled", async () => {
    tabs = [{ id: 42, url: "https://chatgpt.com/c/conv-123", title: "ChatGPT" }];
    stored.polylogueReceiverPairing = {
      state: "online",
      receiver_id: "rx-throttle",
      api_schema: "polylogue-browser-capture/v1",
      endpoint: "http://127.0.0.1:8875",
    };
    const now = vi.spyOn(Date, "now");
    now.mockReturnValue(100000);
    await sendRuntimeMessage({ type: "polylogue.captureSupportedTabs", reason: "popup_sync_open_tabs" });
    now.mockReturnValue(105000);
    globalThis.fetch = vi.fn(async (url) => {
      if (String(url).endsWith("/v1/status")) {
        return responseJson({
          ok: true,
          receiver_id: "rx-throttle",
          api_schema: "polylogue-browser-capture/v1",
        });
      }
      return responseJson({
        provider: "chatgpt",
        provider_session_id: "conv-123",
        state: "missing",
        captured: false,
      });
    });

    activatedListener({ tabId: 42 });

    await vi.waitFor(() => expect(stored.polylogueState?.active_page_state).toBe("conversation"));
    expect(globalThis.chrome.tabs.sendMessage).toHaveBeenCalledTimes(1);
    await vi.waitFor(() => expect(stored.polylogueConversationTimeline["chatgpt:conv-123"]?.[0]).toMatchObject({
      event: "held_with_reason",
      reason: "auto_capture_missing",
      detail: "background_capture_throttled",
    }));
  });

  it("refreshes receiver status for supported pages without a conversation id", async () => {
    expect(updatedListener).toBeTypeOf("function");
    tabs = [{ id: 42, url: "https://chatgpt.com/", title: "ChatGPT" }];
    globalThis.fetch = vi.fn(async (url, options) => {
      fetchCalls.push({ url, options });
      return responseJson({ ok: true, active: true }, { requestId: "status-1" });
    });

    updatedListener(42, { status: "complete" }, tabs[0]);

    await vi.waitFor(() => expect(stored.polylogueState?.active_page_state).toBe("supported_no_session"));
    expect(fetchCalls[0].url).toBe("http://127.0.0.1:8875/v1/status");
    expect(stored.polylogueState.provider).toBe("chatgpt");
    expect(stored.polylogueState.last_receiver_request_id).toBe("status-1");
    expect(globalThis.chrome.scripting.executeScript).not.toHaveBeenCalled();
    expect(globalThis.chrome.tabs.sendMessage).not.toHaveBeenCalled();
  });

  it("injects capture scripts into existing provider tabs on explicit sync", async () => {
    tabs = [{ id: 42, url: "https://chatgpt.com/c/conv-sync", title: "ChatGPT" }];
    await sendRuntimeMessage({ type: "polylogue.captureSupportedTabs", reason: "popup_sync_open_tabs" });

    await vi.waitFor(() => expect(globalThis.chrome.tabs.sendMessage).toHaveBeenCalledTimes(1));

    expect(globalThis.chrome.scripting.executeScript).toHaveBeenCalledWith({
      target: { tabId: 42 },
      files: ["src/content/asset_stream.js", "src/content/chatgpt_bridge.js"],
      world: "MAIN",
    });
    expect(globalThis.chrome.scripting.executeScript).toHaveBeenCalledWith({
      target: { tabId: 42 },
      files: [
        "src/common.js",
        "src/operator_status.js",
        "src/content/message_layer.js",
        "src/content/ambient_surface.js",
        "src/content/asset_stream.js",
        "src/content/chatgpt.js",
      ],
    });
    expect(globalThis.chrome.tabs.sendMessage).toHaveBeenCalledWith(42, {
      type: "polylogue.capturePage",
      reason: "popup_sync_open_tabs",
    });
    expect(stored.polylogueState.online).toBe(true);
    expect(stored.polylogueState.captured).toBe(true);
    expect(stored.polylogueState.last_receiver_request_id).toBe("capture-request-1");
  });

  it("waits for a valid slow capture instead of timing it out", async () => {
    vi.useFakeTimers();
    tabs = [{ id: 42, url: "https://chatgpt.com/c/conv-slow", title: "ChatGPT" }];
    let complete;
    globalThis.chrome.tabs.sendMessage = vi.fn(() => new Promise((resolve) => { complete = resolve; }));
    let settled = false;
    const responsePromise = sendRuntimeMessage({ type: "polylogue.captureSupportedTabs", reason: "popup_sync_open_tabs" })
      .then((result) => { settled = true; return result; });
    await vi.advanceTimersByTimeAsync(40_000);
    expect(settled).toBe(false);
    complete({ ok: true, captureResult: { provider: "chatgpt", provider_session_id: "conv-slow" } });
    expect(await responsePromise).toEqual({ ok: true });
    expect(stored.polylogueCaptureLog[0].ok).toBe(true);
  });

  it("injects the Grok native bridge and content script for open grok.com tabs", async () => {
    tabs = [{ id: 77, url: "https://grok.com/c/1f9de430-6505-4d43-935b-ec0dd1c13222", title: "Grok" }];

    await sendRuntimeMessage({ type: "polylogue.captureSupportedTabs", reason: "popup_sync_open_tabs" });

    await vi.waitFor(() => expect(globalThis.chrome.tabs.sendMessage).toHaveBeenCalledTimes(1));

    expect(globalThis.chrome.scripting.executeScript).toHaveBeenCalledWith({
      target: { tabId: 77 },
      files: ["src/content/asset_stream.js", "src/content/grok_bridge.js"],
      world: "MAIN",
    });
    expect(globalThis.chrome.scripting.executeScript).toHaveBeenCalledWith({
      target: { tabId: 77 },
      files: ["src/content/asset_stream.js", "src/common.js", "src/content/grok.js"],
    });
  });

  // Grok's native REST capture (polylogue Grok native-capture upgrade,
  // 2026-07-31) is only reachable from grok.com itself -- x.com's embedded
  // Grok surface is served through X's own API, not grok.com's
  // /rest/app-chat/* this bridge calls. The DOM-only fallback that used to
  // give x.com/twitter.com tabs a (lossy) capture path was removed in the
  // same change, so those tabs now correctly get no injection at all rather
  // than a script that would silently produce nothing.
  it("does not inject any Grok capture script for x.com/twitter.com tabs", async () => {
    tabs = [{ id: 78, url: "https://x.com/i/grok", title: "Grok" }];

    await sendRuntimeMessage({ type: "polylogue.captureSupportedTabs", reason: "popup_sync_open_tabs" });

    expect(globalThis.chrome.scripting.executeScript).not.toHaveBeenCalled();
    expect(globalThis.chrome.tabs.sendMessage).not.toHaveBeenCalled();
  });
});

describe("capture retry queue", () => {
  beforeEach(async () => {
    vi.restoreAllMocks();
    vi.useRealTimers();
    await loadBackground();
  });

  it("queues a capture for retry when the receiver is unreachable, sets a badge, then drains on the next alarm", async () => {
    let captureCalls = 0;
    globalThis.fetch = vi.fn(async (url, options) => {
      fetchCalls.push({ url });
      if (String(url).endsWith("/v1/status")) {
        return responseJson({
          ok: true,
          receiver_id: "rx-retry",
          api_schema: "polylogue-browser-capture/v1",
        });
      }
      captureCalls += 1;
      if (captureCalls === 1) throw new TypeError("Failed to fetch");
      return captureReceipt({
        ok: true,
        provider: "chatgpt",
        provider_session_id: "conv-9",
        state: "spooled_only",
        artifact_ref: "chatgpt/conv-9.json",
      }, options);
    });

    const envelope = {
      session: {
        provider: "chatgpt",
        provider_session_id: "conv-9",
        provider_meta: { capture_fidelity: "native_full" },
        turns: [{ role: "user" }, { role: "assistant" }],
      },
    };

    tabs = [
      { id: 1, url: "https://chatgpt.com/c/conv-active", active: true },
      { id: 2, url: "https://chatgpt.com/c/conv-9", active: false },
    ];
    stored.polylogueState = { online: true, provider: "chatgpt", provider_session_id: "conv-active", archive_state: { state: "archived" } };
    stored.polylogueReceiverPairing = {
      state: "online", receiver_id: "rx-retry", api_schema: "polylogue-browser-capture/v1",
    };
    const response = await sendRuntimeMessage(
      { type: "polylogue.capture", envelope, reason: "content_script_capture" },
      { tab: tabs[1] },
    );

    expect(response).toEqual({ ok: false, queued: true, error: "Failed to fetch", receiver_request_id: null });
    expect(await deliveryEntries()).toHaveLength(1);
    expect((await deliveryEntries())[0].summary.providerSessionId).toBe("conv-9");
    expect((await deliveryEntries())[0].attempts).toBe(0);
    expect(stored.polylogueConversationTimeline["chatgpt:conv-9"][0]).toMatchObject({
      event: "held_with_reason",
      detail: "capture_queued_for_retry",
    });
    expect(globalThis.chrome.alarms.create).toHaveBeenCalledWith(
      "polylogueCaptureRetry",
      expect.objectContaining({ periodInMinutes: 1 }),
    );
    const lastBadgeCall = globalThis.chrome.action.setBadgeText.mock.calls.at(-1);
    expect(lastBadgeCall[0]).toEqual({ text: "1" });

    // Force the queued entry's backoff window to be due, then simulate the
    // retry alarm firing (real Chrome would deliver this on its own timer).
    await makeDeliveriesDue();
    expect(alarmListener).toBeTypeOf("function");
    alarmListener({ name: "polylogueCaptureRetry" });

    await vi.waitFor(async () => expect(await deliveryEntries()).toHaveLength(0));
    expect(captureCalls).toBe(2);
    expect(globalThis.chrome.alarms.clear).toHaveBeenCalledWith("polylogueCaptureRetry");
    expect(stored.polylogueCaptureLog[0].reason).toBe("capture_retry_drained");
    expect(stored.polylogueState.captured).toBeUndefined();
    expect(stored.polylogueState.provider_session_id).toBe("conv-active");
    expect(stored.polylogueState.archive_state).toEqual({ state: "archived" });
    expect(stored.polylogueSessionLedger["chatgpt:conv-9"].archive_state).toEqual({ state: "spooled_only" });
    expect(stored.polylogueConversationTimeline["chatgpt:conv-9"][0]).toMatchObject({
      event: "captured",
      reason: "capture_retry_drained",
      detail: "spooled_only",
    });
  });

  it("keeps concurrent retry enqueues instead of overwriting one capture", async () => {
    globalThis.fetch = vi.fn(async () => {
      throw new TypeError("Failed to fetch");
    });
    const capture = (sessionId) => sendRuntimeMessage({
      type: "polylogue.capture",
      envelope: { session: { provider: "chatgpt", provider_session_id: sessionId, turns: [] } },
    });

    await Promise.all([capture("conv-concurrent-a"), capture("conv-concurrent-b")]);

    expect((await deliveryEntries()).map((entry) => entry.summary.providerSessionId)).toEqual([
      "conv-concurrent-a",
      "conv-concurrent-b",
    ]);
  });

  it("retains and delivers a retry body beyond the former 40 MiB queue budget", async () => {
    const envelope = { session: { provider: "chatgpt", provider_session_id: "conv-oversized",
      turns: [{ text: "x".repeat(43_000_000) }] } };
    globalThis.fetch = vi.fn(async () => { throw new TypeError("Failed to fetch"); });
    const response = await sendRuntimeMessage({ type: "polylogue.capture", envelope });
    expect(response.queued).toBe(true);
    expect((await deliveryEntries())).toHaveLength(1);
    expect((await deliveryEntries())[0].envelope).toBeUndefined();
    const store = new IndexedDbBackfillStore();
    const staging = new CaptureStaging(globalThis.navigator.storage, store);
    expect(JSON.parse(await (await staging.file((await deliveryEntries())[0].body_ref)).text()).session).toEqual({ ...envelope.session, turns: [{ ...envelope.session.turns[0], ordinal: 0 }] });
    await makeDeliveriesDue();
    globalThis.fetch = vi.fn(async (_url, options) => {
      expect(JSON.parse(await options.body.text()).session).toEqual({ ...envelope.session, turns: [{ ...envelope.session.turns[0], ordinal: 0 }] });
      return responseJson({ ok: true, provider: "chatgpt", provider_session_id: "conv-oversized" });
    });
    expect(await sendRuntimeMessage({ type: "polylogue.retryCaptureQueue" })).toMatchObject({ drained: 1, remaining: 0 });
    expect(await deliveryEntries()).toEqual([]);
  });

  it("reports storage admission failure without dropping an earlier retry", async () => {
    globalThis.fetch = vi.fn(async () => { throw new TypeError("offline"); });
    const capture = (id) => sendRuntimeMessage({ type: "polylogue.capture",
      envelope: { session: { provider: "chatgpt", provider_session_id: id, turns: [] } } });
    await capture("kept");
    const originalPut = IDBObjectStore.prototype.put;
    const failing = vi.spyOn(IDBObjectStore.prototype, "put").mockImplementation(function (...args) {
      if (this.name === "queue" && args[0]?.delivery_kind === "foreground") throw new globalThis.DOMException("storage exhausted", "QuotaExceededError");
      return originalPut.apply(this, args);
    });
    const refused = await capture("unretained");
    expect(refused).toMatchObject({ ok: false, error: "capture_staging_quota_exceeded" });
    expect(refused.queued).not.toBe(true);
    failing.mockRestore();
    expect((await deliveryEntries()).map((entry) => entry.summary.providerSessionId)).toEqual(["kept"]);
    expect(stored.polylogueConversationTimeline["chatgpt:unretained"][0]).toMatchObject({ event: "held_with_reason" });
  });

  it("keeps admitted bodies when the derived status cache fails", async () => {
    globalThis.fetch = vi.fn(async () => { throw new TypeError("offline"); });
    await vi.waitFor(() => expect(stored.polylogueCaptureQueue).toBeDefined());
    const original = globalThis.chrome.storage.local.set;
    globalThis.chrome.storage.local.set = vi.fn(async (patch) => {
      if (patch.polylogueCaptureQueue) throw new globalThis.DOMException("cache exhausted", "QuotaExceededError");
      return original(patch);
    });
    const response = await sendRuntimeMessage({ type: "polylogue.capture",
      envelope: { session: { provider: "chatgpt", provider_session_id: "kept-without-cache", turns: [] } } });
    expect(response).toMatchObject({ ok: false, queued: true });
    expect(await sendRuntimeMessage({ type: "polylogue.getCaptureQueue" })).toMatchObject({
      entries: [expect.objectContaining({ summary: expect.objectContaining({ providerSessionId: "kept-without-cache" }) })],
    });
    expect(globalThis.chrome.alarms.create).toHaveBeenCalledWith("polylogueCaptureRetry", expect.any(Object));
    expect(stored.polylogueDebugLog).toContainEqual(expect.objectContaining({ stage: "capture_queue_cache_failed" }));
    globalThis.chrome.storage.local.set = original;
  });

  it("holds acquired evidence after a later non-retryable receiver rejection", async () => {
    let callCount = 0;
    globalThis.fetch = vi.fn(async () => {
      callCount += 1;
      if (callCount === 1) throw new TypeError("Failed to fetch");
      return responseJson({ error: "invalid capture" }, { ok: false, status: 400 });
    });
    await sendRuntimeMessage({
      type: "polylogue.capture",
      envelope: { session: { provider: "chatgpt", provider_session_id: "conv-retry-rejected", turns: [] } },
    });
    await makeDeliveriesDue();

    alarmListener({ name: "polylogueCaptureRetry" });

    await vi.waitFor(async () => expect((await deliveryEntries())[0]).toMatchObject({ held: true, next_attempt_at: null }));
    expect(await deliveryEntries()).toHaveLength(1);
    expect((await new CaptureStaging(globalThis.navigator.storage).file((await deliveryEntries())[0].body_ref)).size).toBeGreaterThan(0);
    expect(stored.polylogueConversationTimeline["chatgpt:conv-retry-rejected"][0]).toMatchObject({
      event: "held_with_reason",
      detail: "capture_rejected",
    });
    expect(stored.polylogueCaptureLog[0].reason).toBe("capture_retry_rejected");
  });

  it("holds a client-rejected capture visibly without scheduling a retry", async () => {
    globalThis.fetch = vi.fn(async () => responseJson({ error: "invalid_envelope" }, { ok: false, status: 400 }));

    const response = await sendRuntimeMessage({
      type: "polylogue.capture",
      envelope: { session: { provider: "chatgpt", provider_session_id: "conv-1" } },
    });

    expect(response).toEqual({ ok: false, error: "invalid_envelope", outcome: null, retry_after_seconds: null, receiver_request_id: "receiver-request-1" });
    expect((await deliveryEntries())[0]).toMatchObject({ held: true, last_error: "invalid_envelope" });
    expect((await deliveryEntries())[0].next_attempt_at).toBeNull();
  });

  it("retains every 503 retry beyond the former 20-entry queue cap", async () => {
    globalThis.fetch = vi.fn(async () => responseJson({ error: "unavailable" }, { ok: false, status: 503 }));

    for (let i = 0; i < 22; i += 1) {
      await sendRuntimeMessage({
        type: "polylogue.capture",
        envelope: { session: { provider: "chatgpt", provider_session_id: `conv-${i}` } },
      });
    }

    expect(await deliveryEntries()).toHaveLength(22);
    expect(await sendRuntimeMessage({ type: "polylogue.getCaptureQueue" })).toMatchObject({ total: 22, dropped_count: 0 });
    expect(stored.polylogueCaptureQueue?.entries).toBeUndefined();
    expect((await deliveryEntries())[0].summary.providerSessionId).toBe("conv-0");
    expect((await deliveryEntries()).at(-1).summary.providerSessionId).toBe("conv-21");
  });

  it("converts original inline inputs despite derived cache failure without losing evidence or pause state", async () => {
    const queue = { version: 2, dropped_count: 0, entries: ["first", "second"].map((id) => ({
      id, held: true, attempts: 2, queued_at: "2026-01-01T00:00:00Z",
      envelope: { session: { provider: "chatgpt", provider_session_id: id, turns: [{ text: `retained ${id}` }] } },
    })) };
    const settings = { enabled: true, automatic_capture_enabled: false, disabled_sites: [] };
    let allowPublication = false;
    await loadBackground({ polylogueCaptureQueue: globalThis.structuredClone(queue), polylogueAmbientSettings: settings }, () => {
      const publish = globalThis.chrome.storage.local.set.getMockImplementation();
      globalThis.chrome.storage.local.set.mockImplementation(async (patch) => {
        if (patch.polylogueCaptureQueue?.version === 3 && !allowPublication) {
          throw new Error("synthetic_publication_interruption");
        }
        return publish(patch);
      });
    });
    expect(await sendRuntimeMessage({ type: "polylogue.getCaptureQueue" })).toMatchObject({ ok: true, total: 2 });
    expect(stored.polylogueCaptureQueue.entries.every(entry => entry.envelope === undefined)).toBe(true);
    expect(stored.polylogueAmbientSettings).toEqual(settings);
    allowPublication = true;
    const published = await deliveryEntries();
    expect(published.map((entry) => entry.id)).toEqual(["first", "second"]);
    const staging = new CaptureStaging(globalThis.navigator.storage);
    const bodies = await Promise.all(published.map(async (entry) => (await staging.file(entry.body_ref)).text()));
    const writes = globalThis.navigator.storage.writes.length;
    vi.resetModules();
    await import("../src/background.js");
    expect(await sendRuntimeMessage({ type: "polylogue.getCaptureQueue" })).toMatchObject({ ok: true, total: 2 });
    expect(stored.polylogueCaptureQueue).toEqual({ version: 3, dropped_count: 0 });
    const resumed = await deliveryEntries();
    expect(resumed.map((entry) => [entry.id, entry.delivery_sequence, entry.body_ref])).toEqual(published.map((entry) => [entry.id, entry.delivery_sequence, entry.body_ref]));
    expect(await Promise.all(resumed.map(async (entry) => (await staging.file(entry.body_ref)).text()))).toEqual(bodies);
    expect(globalThis.navigator.storage.writes).toHaveLength(writes);
    expect(stored.polylogueAmbientSettings).toEqual(settings);
  });

  it("queues a capture when receiver physical staging fails", async () => {
    globalThis.fetch = vi.fn(async () => responseJson({ error: "write_failed" }, { ok: false, status: 500 }));

    await sendRuntimeMessage({
      type: "polylogue.capture",
      envelope: { session: { provider: "chatgpt", provider_session_id: "conv-stalled" } },
    });

    expect(await deliveryEntries()).toHaveLength(1);
    expect((await deliveryEntries())[0].summary.providerSessionId).toBe("conv-stalled");
  });

  it("summarizes the retry queue for the popup without leaking full envelope internals", async () => {
    stored.polylogueReceiverPairing = { state: "online", receiver_id: "rx-queue-privacy",
      api_schema: "polylogue-browser-capture/v1", endpoint: "http://127.0.0.1:8875" };
    globalThis.fetch = vi.fn(async (url) => {
      if (String(url).endsWith("/v1/status")) return responseJson({ ok: true, receiver_id: "rx-queue-privacy", api_schema: "polylogue-browser-capture/v1" });
      throw new TypeError("offline");
    });
    await sendRuntimeMessage({
      type: "polylogue.capture",
      envelope: {
        session: { provider: "chatgpt", provider_session_id: "conv-5", turns: [{ role: "user", text: "secret" }] },
      },
    }, { tab: { id: 42, url: "https://chatgpt.com/c/conv-5?share=private" } });

    const response = await sendRuntimeMessage({ type: "polylogue.getCaptureQueue" });

    expect(response.ok).toBe(true);
    expect(response.dropped_count).toBe(0);
    expect(response.entries).toHaveLength(1);
    expect(response.entries[0]).toMatchObject({ provider: "chatgpt", provider_session_id: "conv-5", attempts: 0, tab_origin: "https://chatgpt.com" });
    expect(response.entries[0].envelope).toBeUndefined();
    expect(JSON.stringify(response)).not.toContain("share=private");
    expect(response.entries[0].tab_url).toBeUndefined();
  });

  it("drains the retry queue once a subsequent capture proves the receiver is reachable again", async () => {
    let callCount = 0;
    globalThis.fetch = vi.fn(async (_url, options) => {
      callCount += 1;
      if (callCount === 1) throw new TypeError("Failed to fetch");
      return captureReceipt({ ok: true, provider: "chatgpt", provider_session_id: "conv-7" }, options);
    });

    await sendRuntimeMessage({
      type: "polylogue.capture",
      envelope: { session: { provider: "chatgpt", provider_session_id: "conv-7" } },
    });
    expect(await deliveryEntries()).toHaveLength(1);

    // Make the queued entry due, then drive a second capture that succeeds —
    // its success should trigger a queue drain as a side effect.
    await makeDeliveriesDue();
    await sendRuntimeMessage({
      type: "polylogue.capture",
      envelope: { session: { provider: "chatgpt", provider_session_id: "conv-8" } },
    });

    await vi.waitFor(async () => expect(await deliveryEntries()).toHaveLength(0));
  });
});

describe("browser action polling", () => {
  beforeEach(async () => {
    vi.restoreAllMocks();
    vi.useRealTimers();
    await loadBackground();
  });

  it("keeps an unpaired alarm wake entirely local", async () => {
    expect(alarmListener).toBeTypeOf("function");

    alarmListener({ name: "polylogueBrowserActionWake" });

    await vi.waitFor(() => expect(fetchCalls).toHaveLength(0));
  });
});

describe("browser action explicit-approval decision (polylogue-yyvg.7)", () => {
  beforeEach(async () => {
    vi.restoreAllMocks();
    vi.useRealTimers();
    await loadBackground({
      polylogueReceiverPairing: {
        state: "online",
        receiver_id: "rx-approval-test",
        api_schema: "polylogue-browser-capture/v1",
      },
    });
  });

  it("posts an explicit approve decision and wakes the poll loop, never auto-declining", async () => {
    let approvalBody = null;
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      fetchCalls.push({ url, options });
      if (String(url).endsWith("/v1/status")) {
        return responseJson({ ok: true, receiver_id: "rx-approval-test", api_schema: "polylogue-browser-capture/v1" });
      }
      if (String(url).endsWith("/v1/browser-actions/action-held-1/approval")) {
        approvalBody = JSON.parse(options.body);
        return responseJson({ action: { action_id: "action-held-1", status: "queued" } });
      }
      if (String(url).includes("/v1/browser-actions?claim_by=")) {
        return responseJson({ actions: [] });
      }
      return responseJson({ error: "unexpected_receiver_request" }, { ok: false, status: 500 });
    });

    const response = await sendRuntimeMessage({
      type: "polylogue.browserActions.approval",
      actionId: "action-held-1",
      decision: "approve",
    });

    expect(response.ok).toBe(true);
    expect(response.action).toMatchObject({ action_id: "action-held-1", status: "queued" });
    expect(approvalBody).toMatchObject({ decision: "approve" });
    expect(approvalBody.extension_instance_id).toBeTruthy();
    // Approving must never send a decline, and it should wake the claim loop
    // so the now-queued action is picked up without waiting for the next alarm.
    expect(approvalBody.decision).not.toBe("decline");
    await vi.waitFor(() => expect(
      fetchCalls.some((call) => String(call.url).includes("/v1/browser-actions?claim_by=")),
    ).toBe(true));
  });

  it("posts an explicit decline decision without ever polling for a claim", async () => {
    let approvalBody = null;
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      fetchCalls.push({ url, options });
      if (String(url).endsWith("/v1/status")) {
        return responseJson({ ok: true, receiver_id: "rx-approval-test", api_schema: "polylogue-browser-capture/v1" });
      }
      if (String(url).endsWith("/v1/browser-actions/action-held-2/approval")) {
        approvalBody = JSON.parse(options.body);
        return responseJson({ action: { action_id: "action-held-2", status: "cancelled" } });
      }
      return responseJson({ error: "unexpected_receiver_request" }, { ok: false, status: 500 });
    });

    const response = await sendRuntimeMessage({
      type: "polylogue.browserActions.approval",
      actionId: "action-held-2",
      decision: "decline",
    });

    expect(response.ok).toBe(true);
    expect(response.action).toMatchObject({ action_id: "action-held-2", status: "cancelled" });
    expect(approvalBody).toMatchObject({ decision: "decline" });
    expect(fetchCalls.some((call) => String(call.url).includes("claim_by="))).toBe(false);
  });
});

describe("provider-neutral browser action worker", () => {
  beforeEach(async () => {
    vi.restoreAllMocks();
    vi.useRealTimers();
    await loadBackground({
      polylogueReceiverPairing: {
        state: "online",
        receiver_id: "rx-action-test",
        api_schema: "polylogue-browser-capture/v1",
      },
    });
  });

  it("submits in an inactive provider tab and records an owner-bound exact receipt", async () => {
    const action = {
      action_id: "action-1",
      receiver_id: "rx-action-test",
      provider: "chatgpt",
      operation: "conversation.create",
      target: { conversation_id: "new", conversation_url: null, project_ref: null },
      text: "Describe the requested implementation.",
      attachments: [],
      presentation: {
        surface: "chat",
        model_slug: "gpt-5-6-pro",
        model_label: "GPT-5.6 Sol",
        effort_label: "Pro",
      },
      submit_policy: "submit_once",
      status: "leased",
    };
    let claimed = false;
    const updates = [];
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      fetchCalls.push({ url, options });
      if (String(url).endsWith("/v1/status")) {
        return responseJson({ ok: true, receiver_id: "rx-action-test", api_schema: "polylogue-browser-capture/v1" });
      }
      if (String(url).includes("/v1/browser-actions?claim_by=")) {
        if (claimed) return responseJson({ actions: [] });
        claimed = true;
        return responseJson({ actions: [action] });
      }
      if (String(url).endsWith("/v1/browser-actions/action-1/events")) {
        const update = JSON.parse(options.body);
        updates.push(update);
        return responseJson({ action: { ...action, ...update } });
      }
      return responseJson({ error: "unexpected_receiver_request" }, { ok: false, status: 500 });
    });
    globalThis.chrome.scripting.executeScript = mockPageScript(async () => [{ result: {
      ok: true,
      outcome: "submitted",
      provider_conversation_id: "conversation-1",
      provider_conversation_url: "https://chatgpt.com/c/conversation-1",
      provider_turn_id: "user-turn-1",
      observed_surface: "Chat",
      observed_model: "GPT-5.6 Sol",
      observed_effort: "Pro",
      observed_project_ref: null,
      provider_evidence: { current_node: "assistant-running" },
    } }]);

    alarmListener({ name: "polylogueBrowserActionWake" });
    await vi.waitFor(() => expect(updates.at(-1)?.outcome).toBe("submitted"));

    const owner = new globalThis.URL(fetchCalls.find((call) => String(call.url).includes("claim_by="))?.url)
      .searchParams.get("claim_by");
    expect(updates[0]).toMatchObject({ outcome: "progress", phase: "submit_intent" });
    expect(updates.at(-1).receipt).toMatchObject({
      extension_instance_id: owner,
      provider_conversation_id: "conversation-1",
      provider_turn_id: "user-turn-1",
    });
    expect(globalThis.chrome.tabs.create).toHaveBeenCalledWith({
      url: "https://chatgpt.com/",
      active: false,
    });
    expect(globalThis.chrome.tabs.remove).toHaveBeenCalledWith(99);
  });

  it("renews the receiver lease while provider execution remains in flight", async () => {
    const action = {
      action_id: "action-slow",
      receiver_id: "rx-action-test",
      provider: "chatgpt",
      operation: "conversation.create",
      target: { conversation_id: "new", conversation_url: null, project_ref: null },
      text: "Perform a slow provider operation.",
      attachments: [],
      presentation: {
        surface: "chat",
        model_slug: "gpt-5-6-pro",
        model_label: "GPT-5.6 Sol",
        effort_label: "Pro",
      },
      submit_policy: "submit_once",
      status: "leased",
    };
    const updates = [];
    let claimed = false;
    let heartbeat = null;
    let finishExecution = null;
    vi.spyOn(globalThis, "setInterval").mockImplementation((callback) => {
      heartbeat = callback;
      return 123;
    });
    const clearIntervalSpy = vi.spyOn(globalThis, "clearInterval").mockImplementation(() => undefined);
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      if (String(url).endsWith("/v1/status")) {
        return responseJson({ ok: true, receiver_id: "rx-action-test", api_schema: "polylogue-browser-capture/v1" });
      }
      if (String(url).includes("/v1/browser-actions?claim_by=")) {
        if (claimed) return responseJson({ actions: [] });
        claimed = true;
        return responseJson({ actions: [action] });
      }
      if (String(url).endsWith("/v1/browser-actions/action-slow/events")) {
        const update = JSON.parse(options.body);
        updates.push(update);
        return responseJson({ action: { ...action, ...update } });
      }
      return responseJson({ error: "unexpected_receiver_request" }, { ok: false, status: 500 });
    });
    globalThis.chrome.scripting.executeScript = mockPageScript(async (details) => {
      if (!details.func) return undefined;
      return new Promise((resolve) => {
        finishExecution = () => resolve([{ result: {
          ok: true,
          outcome: "submitted",
          provider_conversation_id: "conversation-slow",
          provider_conversation_url: "https://chatgpt.com/c/conversation-slow",
          provider_turn_id: "user-turn-slow",
          observed_surface: "Chat",
          observed_model: "GPT-5.6 Sol",
          observed_effort: "Pro",
          observed_project_ref: null,
          provider_evidence: { current_node: "assistant-running" },
        } }]);
      });
    });

    alarmListener({ name: "polylogueBrowserActionWake" });
    await vi.waitFor(() => expect(heartbeat).toBeTypeOf("function"));
    heartbeat();
    await vi.waitFor(() => expect(updates).toEqual(expect.arrayContaining([
      expect.objectContaining({
        outcome: "progress",
        phase: "submit_intent",
        detail: "renewed browser action lease during provider execution",
      }),
    ])));
    expect(finishExecution).toBeTypeOf("function");
    finishExecution();
    await vi.waitFor(() => expect(updates.at(-1)?.outcome).toBe("submitted"));
    expect(clearIntervalSpy).toHaveBeenCalledWith(123);
  });

  it("honors structured provider Retry-After after the submit boundary fails closed", async () => {
    const action = {
      action_id: "action-rate",
      receiver_id: "rx-action-test",
      provider: "chatgpt",
      operation: "conversation.create",
      target: { conversation_id: "new", conversation_url: null, project_ref: null },
      text: "Harmless rate-limit fixture.",
      attachments: [],
      presentation: { surface: "chat", model_slug: "gpt-5-6-pro", model_label: "GPT-5.6 Sol", effort_label: "Pro" },
      submit_policy: "submit_once",
      status: "leased",
    };
    const updates = [];
    let claimed = false;
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      if (String(url).endsWith("/v1/status")) {
        return responseJson({ ok: true, receiver_id: "rx-action-test", api_schema: "polylogue-browser-capture/v1" });
      }
      if (String(url).includes("/v1/browser-actions?claim_by=")) {
        if (claimed) return responseJson({ actions: [] });
        claimed = true;
        return responseJson({ actions: [action] });
      }
      if (String(url).endsWith("/v1/browser-actions/action-rate/events")) {
        updates.push(JSON.parse(options.body));
        return responseJson({ action });
      }
      return responseJson({ error: "unexpected" }, { ok: false, status: 500 });
    });
    globalThis.chrome.scripting.executeScript = mockPageScript(async () => [{ result: {
      ok: false,
      detail: "provider response http_429",
      retry_after_seconds: 75,
      submission_may_have_occurred: false,
    } }]);

    alarmListener({ name: "polylogueBrowserActionWake" });
    await vi.waitFor(() => expect(updates.at(-1)?.outcome).toBe("rate_limited"));
    expect(updates.at(-1)).toMatchObject({ retry_after_seconds: 75, phase: "provider_action_failed" });
  });

  it("records a provider cooldown for a resolved 429 without Retry-After", async () => {
    const action = {
      action_id: "action-rate-bare",
      receiver_id: "rx-action-test",
      provider: "chatgpt",
      operation: "conversation.create",
      target: { conversation_id: "new", conversation_url: null, project_ref: null },
      text: "Harmless rate-limit fixture.",
      attachments: [],
      presentation: { surface: "chat", model_slug: "gpt-5-6-pro", model_label: "GPT-5.6 Sol", effort_label: "Pro" },
      submit_policy: "submit_once",
      status: "leased",
    };
    const updates = [];
    let claimed = false;
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      if (String(url).endsWith("/v1/status")) {
        return responseJson({ ok: true, receiver_id: "rx-action-test", api_schema: "polylogue-browser-capture/v1" });
      }
      if (String(url).includes("/v1/browser-actions?claim_by=")) {
        if (claimed) return responseJson({ actions: [] });
        claimed = true;
        return responseJson({ actions: [action] });
      }
      if (String(url).endsWith("/v1/browser-actions/action-rate-bare/events")) {
        updates.push(JSON.parse(options.body));
        return responseJson({ action });
      }
      return responseJson({ error: "unexpected" }, { ok: false, status: 500 });
    });
    globalThis.chrome.scripting.executeScript = mockPageScript(async () => [{ result: {
      ok: false,
      detail: "provider response http_429",
      retry_after_seconds: null,
      submission_may_have_occurred: false,
    } }]);

    alarmListener({ name: "polylogueBrowserActionWake" });
    await vi.waitFor(() => expect(updates.at(-1)?.outcome).toBe("rate_limited"));
    // Red if the transport wrapper records a cooldown only when the resolved
    // failure carried a Retry-After: the next operation would reach the
    // provider immediately.
    expect(stored.polylogueCaptureFreshnessQueue?.provider_cooldowns?.chatgpt).toBeGreaterThan(Date.now());
  });

  it("streams an attachment above 16 MiB into bounded owned MAIN calls before recording submit intent", async () => {
    const size = 17 * 1024 * 1024 + 3;
    const chunk = new Uint8Array(65536).fill(110);
    const oracle = createHash("sha256");
    for (let left = size; left > 0; left -= chunk.length) oracle.update(chunk.subarray(0, Math.min(left, chunk.length)));
    const item = { attachment_id: "attachment-1", name: "neutral.bin", mime_type: "application/octet-stream", size_bytes: size, sha256: oracle.digest("hex") };
    const action = {
      action_id: "action-stream", receiver_id: "rx-action-test", provider: "chatgpt", operation: "conversation.create",
      target: { conversation_id: "new" }, text: "Neutral streaming fixture", attachments: [item],
      presentation: { surface: "chat", model_slug: "gpt-5-6-pro", model_label: "GPT-5.6 Sol", effort_label: "Pro" },
      submit_policy: "stage_only", status: "leased",
    };
    const updates = [];
    let claimed = false;
    let released = false;
    let downloaded = 0;
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      if (String(url).endsWith("/v1/status")) return responseJson({ ok: true, receiver_id: "rx-action-test", api_schema: "polylogue-browser-capture/v1" });
      if (String(url).includes("/v1/browser-actions?claim_by=")) { const result = claimed ? [] : [action]; claimed = true; return responseJson({ actions: result }); }
      if (String(url).endsWith("/attachments/attachment-1")) return {
        ok: true, body: { getReader: () => ({
          read: async () => {
            if (downloaded === size) return { done: true };
            const value = chunk.subarray(0, Math.min(chunk.length, size - downloaded));
            downloaded += value.length;
            return { done: false, value };
          },
          releaseLock: () => { released = true; }, cancel: vi.fn(),
        }) },
      };
      if (String(url).endsWith("/action-stream/events")) { updates.push(JSON.parse(options.body)); return responseJson({ action }); }
      return responseJson({ error: "unexpected" }, { ok: false, status: 500 });
    });
    const { transferBrowserActionAttachmentInPage } = await import("../src/actions/chatgpt.js");
    const observed = [];
    globalThis.chrome.scripting.executeScript = vi.fn(async (call) => {
      if (call.func.name === "transferBrowserActionAttachmentInPage") {
        observed.push({ command: call.args[2], encodedLength: call.args[5]?.length || 0 });
        return [{ result: transferBrowserActionAttachmentInPage(...call.args) }];
      }
      const state = globalThis.__polylogueBrowserActionAttachments;
      expect(state.entries.get(item.attachment_id).file.size).toBe(size);
      expect(released).toBe(true);
      expect(updates.at(-1)).toMatchObject({ phase: "preparing" });
      return [{ result: { ok: true, outcome: "drafted", provider_evidence: { attachment_count: 1 } } }];
    });
    try {
      alarmListener({ name: "polylogueBrowserActionWake" });
      await vi.waitFor(() => expect(updates.at(-1)?.outcome).toBe("drafted"), { timeout: 10000 });
      expect(downloaded).toBe(size);
      expect(observed.filter((row) => row.command === "append").length).toBe(Math.ceil(size / 65536));
      expect(Math.max(...observed.map((row) => row.encodedLength))).toBeLessThanOrEqual(Math.ceil(65536 / 3) * 4);
      expect(observed.at(-1).command).toBe("discard");
      expect(globalThis.__polylogueBrowserActionAttachments).toBeUndefined();
    } finally { delete globalThis.__polylogueBrowserActionAttachments; }
  });

  it.each(["hash", "read_failure", "http_failure"])("settles reader and page parts before refusing an attachment %s failure", async (failure) => {
    const item = { attachment_id: "attachment-1", name: "neutral.bin", mime_type: "application/octet-stream", size_bytes: 3, sha256: "00".repeat(32) };
    const action = {
      action_id: "action-integrity", receiver_id: "rx-action-test", provider: "chatgpt", operation: "conversation.create",
      target: { conversation_id: "new" }, text: "Neutral integrity fixture", attachments: [item],
      presentation: { surface: "chat", model_slug: "gpt-5-6-pro", model_label: "GPT-5.6 Sol", effort_label: "Pro" },
      submit_policy: "submit_once", status: "leased",
    };
    const updates = [];
    let claimed = false, read = false, cancelled = false, released = false;
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      if (String(url).endsWith("/v1/status")) return responseJson({ ok: true, receiver_id: "rx-action-test", api_schema: "polylogue-browser-capture/v1" });
      if (String(url).includes("/v1/browser-actions?claim_by=")) { const result = claimed ? [] : [action]; claimed = true; return responseJson({ actions: result }); }
      if (String(url).endsWith("/attachments/attachment-1") && failure === "http_failure") return {
        ok: false, status: 429, headers: { get: () => "7" }, body: { cancel: async () => { cancelled = true; } },
      };
      if (String(url).endsWith("/attachments/attachment-1")) return { ok: true, body: { getReader: () => ({
        read: async () => { if (read) { if (failure === "read_failure") throw new Error("protocol_attachment_read_failed"); return { done: true }; } read = true; return { done: false, value: new Uint8Array([97, 98, 99]) }; },
        cancel: async () => { cancelled = true; }, releaseLock: () => { released = true; },
      }) } };
      if (String(url).endsWith("/action-integrity/events")) { updates.push(JSON.parse(options.body)); return responseJson({ action }); }
      return responseJson({ error: "unexpected" }, { ok: false, status: 500 });
    });
    const { transferBrowserActionAttachmentInPage } = await import("../src/actions/chatgpt.js");
    const commands = [];
    globalThis.chrome.scripting.executeScript = vi.fn(async (call) => {
      expect(call.func.name).toBe("transferBrowserActionAttachmentInPage");
      commands.push(call.args[2]);
      return [{ result: transferBrowserActionAttachmentInPage(...call.args) }];
    });
    try {
      alarmListener({ name: "polylogueBrowserActionWake" });
      await vi.waitFor(() => expect(updates.at(-1)?.outcome).toBe(failure === "http_failure" ? "rate_limited" : "provider_drift"));
      expect(released).toBe(failure !== "http_failure");
      expect(cancelled).toBe(failure !== "hash");
      if (failure === "http_failure") expect(updates.at(-1).retry_after_seconds).toBe(7);
      else expect(commands).toContain("append");
      expect(commands).not.toContain("finish");
      expect(commands.at(-1)).toBe("discard");
      expect(globalThis.__polylogueBrowserActionAttachments).toBeUndefined();
      expect(updates.some((entry) => entry.phase === "submit_intent")).toBe(false);
    } finally { delete globalThis.__polylogueBrowserActionAttachments; }
  });

  it("rejects a response larger than the declared attachment and settles its reader", async () => {
    const action = {
      action_id: "action-oversized",
      receiver_id: "rx-action-test",
      provider: "chatgpt",
      operation: "conversation.create",
      target: { conversation_id: "new", conversation_url: null, project_ref: null },
      text: "Harmless bounded-read fixture.",
      attachments: [{
        attachment_id: "attachment-1",
        name: "context.bin",
        mime_type: "application/octet-stream",
        size_bytes: 1,
        sha256: "00".repeat(32),
      }],
      presentation: { surface: "chat", model_slug: "gpt-5-6-pro", model_label: "GPT-5.6 Sol", effort_label: "Pro" },
      submit_policy: "stage_only",
      status: "leased",
    };
    const updates = [];
    let claimed = false;
    let cancelled = false;
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      if (String(url).endsWith("/v1/status")) {
        return responseJson({ ok: true, receiver_id: "rx-action-test", api_schema: "polylogue-browser-capture/v1" });
      }
      if (String(url).includes("/v1/browser-actions?claim_by=")) {
        if (claimed) return responseJson({ actions: [] });
        claimed = true;
        return responseJson({ actions: [action] });
      }
      if (String(url).includes("/attachments/attachment-1")) {
        const body = new globalThis.ReadableStream({
          start(controller) {
            controller.enqueue(new Uint8Array(8 * 1024 * 1024));
            controller.enqueue(new Uint8Array(9 * 1024 * 1024));
          },
          cancel() {
            cancelled = true;
          },
        });
        return new globalThis.Response(body, { status: 200 });
      }
      if (String(url).endsWith("/v1/browser-actions/action-oversized/events")) {
        updates.push(JSON.parse(options.body));
        return responseJson({ action });
      }
      return responseJson({ error: "unexpected" }, { ok: false, status: 500 });
    });

    alarmListener({ name: "polylogueBrowserActionWake" });
    await vi.waitFor(() => expect(updates.at(-1)?.outcome).toBe("provider_drift"));
    expect(updates.at(-1).detail).toContain("protocol_attachment_size_mismatch");
    expect(cancelled).toBe(true);
    const commands = globalThis.chrome.scripting.executeScript.mock.calls.map(([call]) => call.args?.[2]);
    expect(commands).toContain("discard");
    expect(commands).not.toContain(undefined);
  });
});

describe("ambient capture status", () => {
  beforeEach(async () => {
    vi.restoreAllMocks();
    vi.useRealTimers();
    await loadBackground();
  });

  it("records seeing an already-safe chat as an explicit deduplicated no-action decision", async () => {
    tabs = [{ id: 42, url: "https://chatgpt.com/c/conv-safe", title: "Safe", active: true }];
    let clock = Date.now();
    vi.spyOn(Date, "now").mockImplementation(() => clock);
    globalThis.fetch = vi.fn(async () => responseJson({
      provider: "chatgpt",
      provider_session_id: "conv-safe",
      state: "archived",
      captured: true,
    }));

    activatedListener({ tabId: 42 });
    await vi.waitFor(() => expect(stored.polylogueConversationTimeline?.["chatgpt:conv-safe"])
      .toEqual(expect.arrayContaining([expect.objectContaining({ event: "observed_no_action", detail: "already_safe" })])));
    clock += 5_000;
    activatedListener({ tabId: 42 });
    await vi.waitFor(() => expect(globalThis.fetch).toHaveBeenCalledTimes(2));

    const noActionEvents = stored.polylogueConversationTimeline["chatgpt:conv-safe"]
      .filter((event) => event.event === "observed_no_action");
    expect(noActionEvents).toHaveLength(1);
  });

  it("persists a site-level ambient disable without changing the global default", async () => {
    const response = await sendRuntimeMessage({
      type: "polylogue.ambient.configure",
      site_enabled: false,
    }, { tab: { id: 42, url: "https://chatgpt.com/c/conv-ambient" } });

    expect(response).toMatchObject({
      ok: true,
      ambient: { enabled: true, automatic_capture_enabled: true, site_enabled: false, site: "chatgpt.com" },
    });
    expect(stored.polylogueAmbientSettings).toEqual({
      enabled: true,
      automatic_capture_enabled: true,
      disabled_sites: { "chatgpt.com": true },
    });
  });

  it("persists a global automatic-capture circuit breaker and clears its wake", async () => {
    const response = await sendRuntimeMessage({
      type: "polylogue.ambient.configure",
      automatic_capture_enabled: false,
    });

    expect(response).toMatchObject({
      ok: true,
      ambient: { enabled: true, automatic_capture_enabled: false },
    });
    expect(stored.polylogueAmbientSettings).toEqual({
      enabled: true,
      automatic_capture_enabled: false,
      disabled_sites: {},
    });
    expect(globalThis.chrome.alarms.clear).toHaveBeenCalledWith("polylogueCaptureFreshnessWake");
  });
});

describe("receiver health probe", () => {
  it.each([
    ["receiver_identity_mismatch", "pairing_mismatch", "mismatch"],
    ["receiver_authentication_failed", "unauthorized", "unauthorized"],
  ])("classifies native %s without transferring a credential", async (code, status, state) => {
    await loadBackground({ receiverAuthToken: "neutral-retired", neutralUnrelatedSetting: "preserved", polylogueReceiverPairing: {
      receiver_id: "neutral-original", api_schema: "polylogue-browser-capture/v1", state: "online",
    } });
    globalThis.fetch = vi.fn(async (_url, init) => {
      expect(init.receiverId).toBe("neutral-original");
      expect(init.headers.Authorization).toBeUndefined();
      throw new Error(code);
    });
    const result = await sendRuntimeMessage({ type: "polylogue.checkReceiverHealth" });
    expect(result).toMatchObject({ ok: false, status, detail: code, pairing: { state, receiver_id: "neutral-original" } });
    expect(globalThis.fetch).toHaveBeenCalledOnce();
    expect(stored.receiverAuthToken).toBeUndefined();
    expect(stored.neutralUnrelatedSetting).toBe("preserved");
  });
  beforeEach(async () => {
    vi.restoreAllMocks();
    vi.useRealTimers();
    await loadBackground();
  });

  it("reports the receiver as reachable and authorized when /v1/status returns ok:true", async () => {
    globalThis.fetch = vi.fn(async (url) => {
      fetchCalls.push({ url });
      return responseJson({
        ok: true,
        api_schema: "polylogue-browser-capture/v1",
        receiver_id: "rx-health",
      });
    });

    const response = await sendRuntimeMessage({ type: "polylogue.checkReceiverHealth" });

    expect(response).toMatchObject({
      ok: true,
      status: "ok",
      detail: null,
      endpoint: "http://127.0.0.1:8875",
      receiver_request_id: "receiver-request-1",
      pairing: {
        state: "online",
        receiver_id: "rx-health",
        api_schema: "polylogue-browser-capture/v1",
      },
    });
    expect(fetchCalls[0].url).toBe("http://127.0.0.1:8875/v1/status");
  });

  it("preflights paired writes and reuses the short trusted-identity cache", async () => {
    await loadBackground({
      polylogueReceiverPairing: {
        state: "online",
        receiver_id: "rx-stable",
        api_schema: "polylogue-browser-capture/v1",
        endpoint: "http://127.0.0.1:8875",
      },
    });
    globalThis.fetch = vi.fn(async (url, options) => {
      fetchCalls.push({ url });
      if (String(url).endsWith("/v1/status")) {
        return responseJson({
          ok: true,
          api_schema: "polylogue-browser-capture/v1",
          receiver_id: "rx-stable",
        });
      }
      if (String(url).endsWith("/v1/browser-captures")) {
        return captureReceipt({ provider: "chatgpt", provider_session_id: "conv-trusted", state: "spooled_only" }, options);
      }
      return responseJson({ error: "unexpected" }, { ok: false, status: 500 });
    });

    const message = {
      type: "polylogue.capture",
      envelope: { session: { provider: "chatgpt", provider_session_id: "conv-trusted", turns: [{ role: "user" }] } },
    };
    expect(await sendRuntimeMessage(message)).toMatchObject({ ok: true });
    expect(await sendRuntimeMessage(message)).toMatchObject({ ok: true });

    expect(fetchCalls.filter((call) => String(call.url).endsWith("/v1/status"))).toHaveLength(1);
    expect(fetchCalls.filter((call) => String(call.url).endsWith("/v1/browser-captures"))).toHaveLength(2);
  });

  it("fails closed before a paired capture reaches a different receiver", async () => {
    await loadBackground({
      polylogueReceiverPairing: {
        state: "online",
        receiver_id: "rx-expected",
        api_schema: "polylogue-browser-capture/v1",
        endpoint: "http://127.0.0.1:8875",
      },
    });
    globalThis.fetch = vi.fn(async (url) => {
      fetchCalls.push({ url });
      return responseJson({
        ok: true,
        api_schema: "polylogue-browser-capture/v1",
        receiver_id: "rx-replacement",
      });
    });

    const response = await sendRuntimeMessage({
      type: "polylogue.capture",
      reason: "content_script_capture",
      envelope: { session: { provider: "chatgpt", provider_session_id: "conv-mismatch", turns: [{ role: "user" }] } },
    });

    expect(response).toMatchObject({ ok: false, error: "receiver_pairing_mismatch" });
    expect(fetchCalls).toHaveLength(1);
    expect(String(fetchCalls[0].url)).toMatch(/\/v1\/status$/);
    expect(stored.polylogueCaptureQueue?.entries || []).toHaveLength(0);
    expect(stored.polylogueReceiverPairing).toMatchObject({
      state: "mismatch",
      receiver_id: "rx-expected",
      observed_receiver_id: "rx-replacement",
    });
    expect(stored.polylogueConversationTimeline["chatgpt:conv-mismatch"][0]).toMatchObject({
      event: "held_with_reason",
      detail: "receiver_pairing_mismatch",
    });
  });

  it("recovers only at the canonical endpoint when the paired identity matches", async () => {
    await loadBackground({
      polylogueReceiverPairing: {
        state: "offline",
        receiver_id: "rx-recover",
        api_schema: "polylogue-browser-capture/v1",
        endpoint: "http://127.0.0.1:8875",
      },
    });
    globalThis.fetch = vi.fn(async (url) => {
      fetchCalls.push({ url });
      if (String(url).startsWith("http://127.0.0.1:8875")) throw new TypeError("old endpoint offline");
      return responseJson({
        ok: true,
        api_schema: "polylogue-browser-capture/v1",
        receiver_id: "rx-recover",
      });
    });

    const response = await sendRuntimeMessage({ type: "polylogue.checkReceiverHealth" });

    expect(response).toMatchObject({
      ok: true,
      status: "recovered",
      endpoint: "http://127.0.0.1:8765",
      recovered_from: "http://127.0.0.1:8875",
    });
    expect(stored.receiverBaseUrl).toBe("http://127.0.0.1:8765");
    expect(fetchCalls.map((call) => call.url)).toEqual([
      "http://127.0.0.1:8875/v1/status",
      "http://127.0.0.1:8765/v1/status",
    ]);
  });

  it("does not adopt the canonical endpoint when its receiver identity differs", async () => {
    await loadBackground({
      polylogueReceiverPairing: {
        state: "offline",
        receiver_id: "rx-expected",
        api_schema: "polylogue-browser-capture/v1",
        endpoint: "http://127.0.0.1:8875",
      },
    });
    globalThis.fetch = vi.fn(async (url) => {
      fetchCalls.push({ url });
      if (String(url).startsWith("http://127.0.0.1:8875")) throw new TypeError("old endpoint offline");
      return responseJson({
        ok: true,
        api_schema: "polylogue-browser-capture/v1",
        receiver_id: "rx-other",
      });
    });

    const response = await sendRuntimeMessage({ type: "polylogue.checkReceiverHealth" });

    expect(response).toMatchObject({ ok: false, status: "unreachable", endpoint: "http://127.0.0.1:8875" });
    expect(stored.receiverBaseUrl).toBe("http://127.0.0.1:8875");
    expect(stored.polylogueReceiverPairing.receiver_id).toBe("rx-expected");
  });

  it("does not auto-recover a deliberate dev-override pairing even when the canonical identity would match (polylogue-jlme.5)", async () => {
    // Reproduces the exact polylogue-jlme.5 incident shape in reverse: an
    // operator has explicitly configured this profile to a dev-loop
    // receiver (dev_override: true). That receiver goes offline. Unlike the
    // ordinary "wandered off" case, this must NOT silently fail over to the
    // canonical endpoint -- the operator chose the dev endpoint on purpose
    // and needs to know loudly that it died, not have the extension quietly
    // reconnect somewhere else mid dev-loop session.
    await loadBackground({
      polylogueReceiverPairing: {
        state: "offline",
        receiver_id: "rx-dev-loop",
        api_schema: "polylogue-browser-capture/v1",
        endpoint: "http://127.0.0.1:8876",
        dev_override: true,
      },
    });
    globalThis.fetch = vi.fn(async (url) => {
      fetchCalls.push({ url });
      if (String(url).startsWith("http://127.0.0.1:8875")) throw new TypeError("dev endpoint offline");
      // The canonical endpoint is live and WOULD satisfy identity+schema
      // match -- this is exactly what proves recovery is suppressed
      // deliberately, not merely because canonical happened to be unreachable.
      return responseJson({
        ok: true,
        api_schema: "polylogue-browser-capture/v1",
        receiver_id: "rx-dev-loop",
      });
    });
    await globalThis.chrome.storage.local.set({ receiverBaseUrl: "http://127.0.0.1:8875" });

    const response = await sendRuntimeMessage({ type: "polylogue.checkReceiverHealth" });

    expect(response).toMatchObject({ ok: false, status: "dev_override_stale", endpoint: "http://127.0.0.1:8875" });
    // Canonical was never even probed.
    expect(fetchCalls.map((call) => call.url)).toEqual(["http://127.0.0.1:8875/v1/status"]);
    expect(stored.receiverBaseUrl).toBe("http://127.0.0.1:8875");
    expect(stored.polylogueReceiverPairing).toMatchObject({
      state: "dev_override_stale",
      dev_override: true,
      receiver_id: "rx-dev-loop",
    });
  });

  it("marks a manually configured non-canonical endpoint as a deliberate dev override", async () => {
    await loadBackground({
      polylogueReceiverPairing: {
        state: "online",
        receiver_id: "rx-canonical",
        api_schema: "polylogue-browser-capture/v1",
        endpoint: "http://127.0.0.1:8765",
      },
    });

    grantedOrigins.add("http://127.0.0.1:8876/*");
    await sendRuntimeMessage({
      type: "polylogue.configureReceiver",
      receiverBaseUrl: "http://127.0.0.1:8876",
      });

    expect(stored.polylogueReceiverPairing).toMatchObject({ dev_override: true });

    // Repointing settings back at the canonical endpoint is an equally
    // explicit act and must clear the flag.
    await sendRuntimeMessage({
      type: "polylogue.configureReceiver",
      receiverBaseUrl: "http://127.0.0.1:8765",
      });

    expect(stored.polylogueReceiverPairing).toMatchObject({ dev_override: false });
  });

  it("refuses a receiver origin the extension does not hold permission for", async () => {
    // polylogue-tztk (leak audit L7): the manifest used to grant
    // http://127.0.0.1/*, which also covers the unauthenticated archive API
    // on 8766, and extension fetches bypass CORS. Anti-vacuity: drop the
    // loopbackOriginIsGranted check in saveReceiverSettings and the settings
    // are stored, making both assertions below red.
    await loadBackground();
    const before = stored.receiverBaseUrl;

    const response = await sendRuntimeMessage({
      type: "polylogue.configureReceiver",
      receiverBaseUrl: "http://127.0.0.1:8766",
      });

    expect(response).toMatchObject({ ok: false, error: "receiver_origin_not_permitted" });
    expect(stored.receiverBaseUrl).toBe(before);
  });

  it("accepts a non-default receiver origin once the operator has granted it", async () => {
    await loadBackground();
    grantedOrigins.add("http://127.0.0.1:8766/*");

    const response = await sendRuntimeMessage({
      type: "polylogue.configureReceiver",
      receiverBaseUrl: "http://127.0.0.1:8766",
      });

    expect(response).toMatchObject({ ok: true });
    expect(stored.receiverBaseUrl).toBe("http://127.0.0.1:8766");
  });

  it("resets only the pairing key and preserves pending work", async () => {
    const queue = { version: 3, dropped_count: 0 };
    await loadBackground({
      polylogueCaptureQueue: queue,
      polylogueReceiverPairing: {
        state: "mismatch",
        receiver_id: "rx-old",
        api_schema: "polylogue-browser-capture/v1",
      },
    });
    const captureStore = new IndexedDbBackfillStore(globalThis.indexedDB);
    const staging = new CaptureStaging(globalThis.navigator.storage, captureStore);
    const delivery = { id: "queued-capture", delivery_kind: "foreground", held: true, queued_at: "2026-01-01T00:00:00Z" };
    const prepared = await staging.prepare({ session: { provider: "chatgpt", provider_session_id: "queued-session", turns: [{ text: "retained acquired evidence" }] } }, null, delivery);
    const before = await captureStore.getDelivery(delivery.id);
    const body = await prepared.body.text();
    globalThis.fetch = vi.fn(async () => responseJson({
      ok: true,
      api_schema: "polylogue-browser-capture/v1",
      receiver_id: "rx-new",
    }));

    const response = await sendRuntimeMessage({ type: "polylogue.receiverPairing.reset" });

    expect(response.pairing).toMatchObject({ state: "online", receiver_id: "rx-new" });
    expect(stored.polylogueCaptureQueue).toEqual(queue);
    expect(await captureStore.getDelivery(delivery.id)).toEqual(before);
    expect(await (await staging.file(prepared.ref)).text()).toBe(body);
  });

  it("reports the receiver as unreachable when the fetch itself fails", async () => {
    globalThis.fetch = vi.fn(async () => {
      throw new TypeError("Failed to fetch");
    });

    const response = await sendRuntimeMessage({ type: "polylogue.checkReceiverHealth" });

    expect(response).toMatchObject({ ok: false, status: "unreachable", detail: "Failed to fetch" });
  });

  it("reports the receiver as unreachable when the response body is not JSON", async () => {
    globalThis.fetch = vi.fn(async () => ({
      headers: { get: vi.fn(() => null) },
      json: vi.fn(async () => {
        throw new Error("not json");
      }),
      ok: true,
      status: 200,
    }));

    const response = await sendRuntimeMessage({ type: "polylogue.checkReceiverHealth" });

    expect(response).toMatchObject({ ok: false, status: "unreachable", detail: "non_json_response" });
  });
});

describe("accepted message identity storage", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  it("drops a version 1 identity cache instead of reading it", async () => {
    // Anti-vacuity: without the startup replacement the snapshot indexes the
    // stale scalar entry as if it were a keyed map and reports its fields as
    // accepted message refs.
    const legacy = {
      message_ref: "chatgpt-export:conv-123:n:m1",
      evidence_ref: "chatgpt/conv-123.json#message:m1",
      fidelity: "native",
    };
    await loadBackground({
      polylogueState: { provider: "chatgpt", provider_session_id: "conv-123" },
      polylogueAcceptedMessageIdentities: { "chatgpt:conv-123": legacy },
    });
    globalThis.fetch = vi.fn(async () => responseJson({ ok: false }, { ok: false, status: 503 }));

    const snapshot = await sendRuntimeMessage(
      { type: "polylogue.missionControl.status", refresh: false },
      { tab: { id: 7, url: "https://chatgpt.com/c/conv-123", title: "conversation" } },
    );

    expect(stored.polylogueAcceptedMessageIdentitiesVersion).toBe(2);
    expect(stored.polylogueAcceptedMessageIdentities).toEqual({});
    expect(snapshot.assertions.accepted_identities).toEqual({});
  });
});
