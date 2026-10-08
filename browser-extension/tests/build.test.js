// Tests for scripts/build.mjs + scripts/validate-manifest.mjs.
//
// These exercise the version sync + Firefox manifest transform + archive
// emission so a future change to the build pipeline cannot silently break
// the release artifact shape.

import { createHash, webcrypto } from "node:crypto";
import { Script } from "node:vm";
import { JSDOM } from "jsdom";
import { memoryOriginStorage } from "./infra/capture-staging.js";
import { checkpointReceiverState } from "./infra/capture-job-checkpoints.js";
import { scheduleFreshnessHint } from "../src/capture/freshness.js";
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { IDBFactory, IDBKeyRange, indexedDB } from "fake-indexeddb";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const __dirname = dirname(fileURLToPath(import.meta.url));
const EXT_ROOT = resolve(__dirname, "..");
const BUILD = join(EXT_ROOT, "scripts", "build.mjs");
const VALIDATE = join(EXT_ROOT, "scripts", "validate-manifest.mjs");
const MANIFEST_PATH = join(EXT_ROOT, "manifest.json");
const PACKAGE_PATH = join(EXT_ROOT, "package.json");

const LOCK_PATH = join(EXT_ROOT, "package-lock.json");
const ORIGINAL_LOCK = readFileSync(LOCK_PATH, "utf8");
const ORIGINAL_MANIFEST = readFileSync(MANIFEST_PATH, "utf8");
const ORIGINAL_PACKAGE = readFileSync(PACKAGE_PATH, "utf8");
const { Headers, Response } = globalThis;

function restore() {
  writeFileSync(MANIFEST_PATH, ORIGINAL_MANIFEST);
  writeFileSync(PACKAGE_PATH, ORIGINAL_PACKAGE);
  writeFileSync(LOCK_PATH, ORIGINAL_LOCK);
}

describe("validate-manifest.mjs", () => {
  it("accepts the committed manifest", () => {
    expect(() => execFileSync("node", [VALIDATE], { stdio: "pipe" })).not.toThrow();
  });

  it("rejects a manifest with an overly broad host permission", async () => {
    const dir = await mkdtemp(join(tmpdir(), "polylogue-ext-validate-"));
    try {
      const broken = JSON.parse(ORIGINAL_MANIFEST);
      broken.host_permissions = ["<all_urls>"];
      const path = join(dir, "manifest.json");
      writeFileSync(path, JSON.stringify(broken, null, 2));
      let threw = false;
      try {
        execFileSync("node", [VALIDATE, path], { stdio: "pipe" });
      } catch {
        threw = true;
      }
      expect(threw).toBe(true);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe("build.mjs", () => {
  afterEach(restore);

  it("synchronizes manifest, package and lock root without changing dependency pins", () => {
    execFileSync("node", [BUILD, "--version", "9.8.7", "--sync-only"], { stdio: "pipe" });
    const manifest = JSON.parse(readFileSync(MANIFEST_PATH, "utf8"));
    const pkg = JSON.parse(readFileSync(PACKAGE_PATH, "utf8"));
    expect(manifest.version).toBe("9.8.7");
    expect(pkg.version).toBe("9.8.7");
    const lock = JSON.parse(readFileSync(LOCK_PATH, "utf8"));
    const original = JSON.parse(ORIGINAL_LOCK);
    expect(lock.version).toBe("9.8.7");
    expect(lock.packages[""].version).toBe("9.8.7");
    delete lock.version; delete original.version;
    delete lock.packages[""].version; delete original.packages[""].version;
    expect(lock).toEqual(original);
  });

  it("strips dev/pre-release suffixes before writing the Chrome version", () => {
    execFileSync("node", [BUILD, "--version", "1.2.3.dev4+gabc", "--sync-only"], { stdio: "pipe" });
    const manifest = JSON.parse(readFileSync(MANIFEST_PATH, "utf8"));
    expect(manifest.version).toBe("1.2.3");
  });
});

describe("build.mjs full archive emission", () => {
  let outDir;
  beforeEach(async () => {
    outDir = await mkdtemp(join(tmpdir(), "polylogue-ext-out-"));
  });
  afterEach(() => {
    rmSync(outDir, { recursive: true, force: true });
    restore();
  });

  it("emits chrome zip + firefox xpi with build-manifest.json", () => {
    execFileSync("node", [BUILD, "--version", "0.9.0", "--out", outDir, "--no-source-sync"], { stdio: "pipe" });
    expect(existsSync(join(outDir, "build-manifest.json"))).toBe(true);
    expect(existsSync(join(outDir, "polylogue-browser-capture-0.9.0-chrome.zip"))).toBe(true);
    expect(existsSync(join(outDir, "polylogue-browser-capture-0.9.0-firefox.xpi"))).toBe(true);
    const summary = JSON.parse(readFileSync(join(outDir, "build-manifest.json"), "utf8"));
    expect(summary.version).toBe("0.9.0");
    expect(summary.firefox_gecko_id).toMatch(/@/);
    const listing = execFileSync(
      "python3",
      ["-c", "import sys,zipfile; print('\\n'.join(zipfile.ZipFile(sys.argv[1]).namelist()))", join(outDir, "polylogue-browser-capture-0.9.0-chrome.zip")],
      { encoding: "utf8" },
    );
    expect(listing).toContain("src/backfill/coordinator.js");
    expect(listing).toContain("src/backfill/providers.js");
    expect(listing).toContain("src/backfill/storage.js");
    expect(listing).toContain("src/backfill/page_transport.js");
  }, 15_000);

  it("executes the packaged service worker fixture without foreground tab activation", async () => {
    const smokeRoot = join(EXT_ROOT, ".cache", `packaged-smoke-${Date.now()}`);
    mkdirSync(smokeRoot, { recursive: true });
    const sourceHashBefore = createHash("sha256").update(readFileSync(MANIFEST_PATH)).update(readFileSync(PACKAGE_PATH)).digest("hex");
    execFileSync("node", [BUILD, "--version", "0.9.0", "--out", smokeRoot, "--no-source-sync"], { stdio: "pipe" });
    const sourceHashAfter = createHash("sha256").update(readFileSync(MANIFEST_PATH)).update(readFileSync(PACKAGE_PATH)).digest("hex");
    expect(sourceHashAfter).toBe(sourceHashBefore);
    const archive = join(smokeRoot, "polylogue-browser-capture-0.9.0-chrome.zip");
    const unpacked = join(smokeRoot, "unpacked");
    execFileSync("python3", ["-c", "import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])", archive, unpacked]);
    let messageListener;
    let alarmListener;
    // The backfill exact-capture path (requirePairedTrustedReceiver,
    // fix(extension): require pairing before provider capture #2983) needs a
    // real receiver pairing, not just a configured base URL + token -- a
    // manually-configured-but-never-paired receiver is a distinct state from
    // the one this fixture exercises. Seed it the same way
    // background.test.js's successful-capture fixtures do.
    let stored = {
      receiverBaseUrl: "http://127.0.0.1:8765",
      receiverAuthToken: "token",
      polylogueReceiverPairing: {
        state: "online",
        receiver_id: "packaged-receiver",
        api_schema: "polylogue-browser-capture/v1",
      },
    };
    let sessionStored = {};
    const pageRequests = [];
    const pageFetchCalls = [];
    const pageToken = "packaged-page-token";
    const pageAccount = "packaged-page-account";
    let replyNativeId = "fixture-1";
    let heldNativeFetch = null;
    const pageDom = new JSDOM("<!doctype html>", { url: "https://chatgpt.com/", runScripts: "outside-only" });
    const pageWindow = pageDom.window;
    Object.defineProperty(pageWindow, "crypto", { configurable: true, value: webcrypto });
    Object.defineProperty(pageWindow, "fetch", { configurable: true, value: vi.fn(async (input, options = {}) => {
        const url = new globalThis.URL(input);
        pageFetchCalls.push({ url, options });
        if (url.pathname === "/api/auth/session") {
          return new Response(JSON.stringify({ accessToken: pageToken, account: { id: pageAccount } }), { headers: { "Content-Type": "application/json" } });
        }
        const headers = new Headers(options.headers);
        if (headers.get("Authorization") !== `Bearer ${pageToken}` || headers.get("ChatGPT-Account-Id") !== pageAccount) {
          return new Response(JSON.stringify({ items: [], total: 0 }), { headers: { "Content-Type": "application/json" } });
        }
        if (url.pathname === "/backend-api/conversations") {
          return new Response(JSON.stringify({ items: [{ id: "fixture-1", update_time: 1780000000 }], total: 1 }), { headers: { "Content-Type": "application/json" } });
        }
        if (heldNativeFetch) {
          const held = heldNativeFetch;
          held.entered(); await held.release;
          options.signal.throwIfAborted();
        }
        return new Response(JSON.stringify({ id: replyNativeId, mapping: { one: { message: { id: "m1", author: { role: "user" }, content: { parts: ["fixture"] } } } } }), { headers: { "Content-Type": "application/json" } });
      }) });
    // Passive/automatic backfill work is observe-only: it must find an
    // already-open, operator-owned provider tab via chrome.tabs.query and
    // never call chrome.tabs.create itself (fix(extension): avoid automatic
    // provider transport tabs, #2974; background.test.js's "never creates a
    // transport tab when passive backfill has no provider page" pins this).
    // Seed one open ChatGPT tab up front so this fixture models that real
    // precondition instead of a permanently empty tab list, and make
    // chrome.tabs.query reflect it dynamically like background.test.js does.
    const ownedTabs = [{ id: 42, url: "https://chatgpt.com/", active: true, status: "complete" }];
    const pageRuntimeListeners = [];
    let admissionGate = null;
    const holdAdmission = async (phase) => {
      if (!admissionGate?.enabled || admissionGate.phase !== phase) return;
      const gate = admissionGate; gate.enabled = false; gate.entered(); await gate.release;
    };
    const tabs = {
      create: vi.fn(async ({ url, active }) => {
        const tab = { id: 77, url, active, status: "complete" };
        ownedTabs.push(tab);
        return tab;
      }),
      get: vi.fn(async (tabId) => { await holdAdmission("tab"); return ownedTabs.find((tab) => tab.id === tabId); }),
      update: vi.fn(),
      remove: vi.fn(),
      query: vi.fn(async () => ownedTabs),
      sendMessage: vi.fn(async (_tabId, message) => {
        if (message.type === "polylogue.capturePage" && admissionGate) admissionGate.enabled = true;
        if (message.type === "polylogue.stagingOwner") await holdAdmission("document");
        return new Promise((resolve) => {
        const handled = pageRuntimeListeners.some((listener) => listener(message, {}, resolve) === true);
        if (!handled) resolve(undefined);
        });
      }),
    };
    globalThis.indexedDB = indexedDB;
    globalThis.IDBKeyRange = IDBKeyRange;
    Object.defineProperty(globalThis, "navigator", { configurable: true, value: { storage: memoryOriginStorage() } });
    globalThis.chrome = {
      action: { setBadgeText: vi.fn(), setBadgeBackgroundColor: vi.fn() },
      alarms: { create: vi.fn(), clear: vi.fn(), onAlarm: { addListener: vi.fn((listener) => { alarmListener = listener; }) } },
      runtime: {
        id: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        getManifest: () => ({ version: "0.9.0" }),
        onInstalled: { addListener: vi.fn() },
        onStartup: { addListener: vi.fn() },
        onMessage: { addListener: vi.fn((listener) => { messageListener = listener; }) },
        sendMessage: (message) => new Promise((resolve) => messageListener(message, {
          tab: { id: 42, url: "https://chatgpt.com/" }, url: "https://chatgpt.com/", documentId: "packaged-document",
        }, resolve)),
      },
      scripting: { executeScript: vi.fn(async (details) => {
        if (details.files) {
          for (const file of details.files) {
            new Script(readFileSync(join(unpacked, file), "utf8"), { filename: file })
              .runInContext(pageDom.getInternalVMContext());
          }
          return [];
        }
        if (!details.args) return [{ result: true, documentId: "packaged-document" }];
        pageRequests.push(details.args[0]);
        const previousWindow = globalThis.window;
        globalThis.window = pageWindow;
        try {
          return [{ result: await details.func(...details.args) }];
        } finally {
          globalThis.window = previousWindow;
        }
      }) },
      storage: {
        local: {
          get: vi.fn(async (defaults) => ({ ...defaults, ...stored })),
          set: vi.fn(async (patch) => { stored = { ...stored, ...patch }; }),
          remove: vi.fn(async (key) => { delete stored[key]; }),
        },
        session: {
          get: vi.fn(async (defaults) => ({ ...defaults, ...sessionStored })),
          set: vi.fn(async (patch) => { sessionStored = { ...sessionStored, ...patch }; }),
          remove: vi.fn(async (key) => { delete sessionStored[key]; }),
        },
      },
      tabs: { ...tabs, onActivated: { addListener: vi.fn() }, onUpdated: { addListener: vi.fn() } },
    };
    Object.defineProperty(pageWindow, "chrome", { configurable: true, value: {
      ...globalThis.chrome,
      runtime: { ...globalThis.chrome.runtime, onMessage: { addListener: (listener) => pageRuntimeListeners.push(listener) } },
    } });
    Object.defineProperty(pageWindow, "postMessage", { configurable: true, value(data) {
      globalThis.queueMicrotask(() => pageWindow.dispatchEvent(new pageWindow.MessageEvent("message", {
        source: pageWindow, origin: pageWindow.location.origin, data,
      })));
    } });
    new Script(readFileSync(join(unpacked, "src", "content", "asset_stream.js"), "utf8"))
      .runInContext(pageDom.getInternalVMContext());
    const fetchCalls = [];
    let receiverPosts = 0;
    const receiverCheckpoints = checkpointReceiverState();
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      fetchCalls.push({ url, options });
      let body;
      if (String(url).endsWith("/v1/browser-captures/capabilities")) {
        return { ok: true, status: 200, headers: { get: (name) => name === "X-Request-ID" ? "packaged-capability" : null }, json: async () => ({ durable_ack_fields: ["receiver_request_id", "content_hash", "submitted_content_hash", "outcome"] }) };
      }
      if (String(url).endsWith("/v1/status")) {
        // requirePairedTrustedReceiver's health probe (ensureTrustedReceiver ->
        // checkReceiverHealth -> GET /v1/status), matching the seeded
        // polylogueReceiverPairing identity above.
        return { ok: true, status: 200, headers: { get: () => null }, json: async () => ({ ok: true, receiver_id: "packaged-receiver", api_schema: "polylogue-browser-capture/v1" }) };
      }
      const receiverPath = new globalThis.URL(url).pathname;
      if (receiverPath === "/v1/archive-state") {
        return { ok: true, status: 200, headers: { get: () => null }, json: async () => ({ sessions: [] }) };
      }
      if (receiverPath === "/v1/capture-jobs/capabilities") {
        return { ok: true, status: 200, headers: { get: () => null }, json: async () => ({
          schema: "polylogue.capture-jobs.capabilities.v1", checkpoint_transport: "canonical-artifact-v1",
          protocol_min: 2,
          protocol_max: 2,
          scope_namespace: "cjs1:packaged-namespace",
        }) };
      }
      if (receiverPath === "/v1/capture-jobs/discover") {
        return { ok: true, status: 200, headers: { get: () => null }, json: async () => ({ jobs: [] }) };
      }
      if (receiverPath === "/v1/capture-jobs") {
        const request = JSON.parse(options.body);
        const scope = request.scope.kind === "invocation" ? { kind: "invocation", resume_capability: "synthetic-packaged-resume" } : request.scope;
        return { ok: true, status: 201, headers: { get: () => null }, json: async () => ({ scope, job: receiverCheckpoints.job({
          job_id: "packaged-capture-job", provider: "chatgpt", scope, revision: 0, lease_generation: 0,
        }) }) };
      }
      if (receiverPath.endsWith("/adopt")) {
        return { ok: true, status: 200, headers: { get: () => null }, json: async () => ({
          job: receiverCheckpoints.job({ job_id: "packaged-capture-job", provider: "chatgpt", revision: 1, lease_generation: 1 }),
          lease: { lease_id: "packaged-lease", generation: 1, proof: "packaged-proof" },
        }) };
      }
      if (receiverPath.endsWith("/update")) {
        return { ok: true, status: 200, headers: { get: () => null }, json: async () => ({
          job: receiverCheckpoints.job({
            job_id: "packaged-capture-job", provider: "chatgpt", revision: 2, lease_generation: 1,
            lease_expires_at: "2026-07-16T10:02:00Z",
          }),
          receipt: { kind: "capture_job_update" },
        }) };
      }
      if (receiverPath.endsWith("/checkpoint")) {
        const checkpoint = await receiverCheckpoints.checkpoint({ job_id: "packaged-capture-job", provider: "chatgpt", revision: 3 }, options);
        return { ok: true, status: 200, headers: { get: () => null }, json: async () => ({
          job: checkpoint.job, receipt: checkpoint.receipt,
        }) };
      }
      const nativeResponse = (payload) => ({ ok: true, status: 200, headers: { get: (name) => name === "X-Request-ID" ? "packaged-native-request" : null }, json: async () => payload });
      if (receiverPath === "/v1/capture-jobs/packaged-capture-job") return nativeResponse({ job: receiverCheckpoints.job({ job_id: "packaged-capture-job" }) });
      if (receiverPath.endsWith("/native/begin")) return nativeResponse({ state: "acquiring" });
      if (receiverPath.endsWith("/native/member")) {
        const descriptor = JSON.parse(options.headers["X-Polylogue-Native"]);
        expect(createHash("sha256").update(globalThis.Buffer.from(await options.body.arrayBuffer())).digest("hex")).toBe(descriptor.sha256);
        return nativeResponse({ sha256: descriptor.sha256, size_bytes: descriptor.size_bytes });
      }
      if (receiverPath.endsWith("/native/prepare")) return nativeResponse({ plan_digest: "sha256:" + "a".repeat(64), summary: { title: null, turn_count: 1, attachment_count: 0, session_kind: "standard", needs_follow_up: true } });
      if (receiverPath.endsWith("/native/plan")) return nativeResponse({ assets: [], after: null });
      if (receiverPath.endsWith("/native/finalize")) return nativeResponse({ sha256: "b".repeat(64), size_bytes: 1234 });
      if (receiverPath.endsWith("/native/publish")) {
        receiverPosts += 1;
        return { ok: true, status: 200, headers: { get: (name) => name === "X-Request-ID" && receiverPosts > 1 ? "packaged-ack" : null },
          json: async () => ({ ok: true, provider: "chatgpt", provider_session_id: "fixture-1", content_hash: JSON.parse(options.body).sha256, submitted_content_hash: JSON.parse(options.body).sha256, outcome: "accepted" }) };
      }
      const contentHash = createHash("sha256").update(globalThis.Buffer.from(await options.body.arrayBuffer())).digest("hex");
      receiverPosts += 1;
      body = receiverPosts === 1
        ? { ok: true, provider: "chatgpt", provider_session_id: "fixture-1", content_hash: contentHash, submitted_content_hash: contentHash, outcome: "accepted" }
        : { ok: true, provider: "chatgpt", provider_session_id: "fixture-1", content_hash: contentHash, submitted_content_hash: contentHash, outcome: "accepted" };
      return { ok: true, status: 200, headers: { get: (name) => name === "X-Request-ID" && receiverPosts > 1 ? "packaged-ack" : null }, json: async () => body };
    });
    const packagedWorkerUrl = `${pathToFileURL(join(unpacked, "src", "background.js")).href}?smoke=${Date.now()}`;
    await import(/* @vite-ignore */ packagedWorkerUrl);
    const { IndexedDbBackfillStore } = await import(/* @vite-ignore */ pathToFileURL(join(unpacked, "src", "backfill", "storage.js")).href);
    const invocationStore = new IndexedDbBackfillStore();
    const send = (message) => new Promise((resolve) => messageListener(message, {}, resolve));
    const started = await send({ type: "polylogue.backfill.start", provider: "chatgpt", cutoff: "2026-01-01T00:00:00Z", policy: { baseCadenceMs: 0 } });
    // Name the refusal: reaching for `started.job.id` on a rejected start
    // reports "undefined has no 'id'" and buries the reason.
    expect(started).toMatchObject({ ok: true });
    await vi.waitFor(() => expect(pageRequests.some((message) => message.operation === "inventory")).toBe(true), { timeout: 15000 });
    const waitForJobIdle = () => vi.waitFor(async () =>
      expect((await invocationStore.getJob(started.job.id)).execution_owner).toBeNull(), { timeout: 15000 });
    for (let inventoryCount = 2; inventoryCount <= 4; inventoryCount += 1) {
      // A request being observed does not mean its owned physical operation
      // has settled. An alarm during that lease joins the current wake.
      await waitForJobIdle();
      alarmListener({ name: `polylogueBackfillWake:${started.job.id}` });
      await vi.waitFor(() => expect(pageRequests.filter((message) => message.operation === "inventory")).toHaveLength(inventoryCount), { timeout: 15000 });
    }
    await waitForJobIdle();
    alarmListener({ name: `polylogueBackfillWake:${started.job.id}` });
    await vi.waitFor(() => expect(receiverPosts).toBe(1), { timeout: 15000 });
    const paused = await send({ type: "polylogue.backfill.status" });
    expect(paused.jobs[0]).toMatchObject({
      status: "paused",
      cooldown_reason: "receiver_contract_incompatible",
      last_error: "receiver_contract_incompatible:missing_receiver_request_id",
    });
    expect(receiverPosts).toBe(1);
    await waitForJobIdle();
    await send({ type: "polylogue.backfill.control", job_id: started.job.id, action: "resume" });
    alarmListener({ name: `polylogueBackfillWake:${started.job.id}` });
    await vi.waitFor(() => expect(receiverPosts).toBe(2), { timeout: 15000 });
    const recovered = await send({ type: "polylogue.backfill.status" });
    expect(recovered.jobs[0].progress.complete).toBe(1);
    expect(pageRequests.filter((message) => message.operation === "conversation")).toHaveLength(1);
    // The whole run rides the pre-seeded operator tab; passive backfill must
    // never materialize its own transport tab or foreground-activate it.
    expect(tabs.create).not.toHaveBeenCalled();
    expect(tabs.update).not.toHaveBeenCalled();
    expect(pageRequests.filter((message) => message.operation !== "identity").map((message) => message.operation))
      .toEqual(["inventory", "inventory", "inventory", "inventory", "conversation"]);
    expect(pageFetchCalls.filter((call) => call.url.pathname === "/api/auth/session").length).toBeGreaterThanOrEqual(5);
    expect(JSON.stringify(pageRequests)).not.toContain(pageToken);
    expect(JSON.stringify(pageRequests)).not.toContain(pageAccount);
    expect(tabs.sendMessage.mock.calls.some(([, message]) => message.type === "polylogue.capturePage")).toBe(false);
    expect(fetchCalls.every((call) => String(call.url).includes("127.0.0.1"))).toBe(true);

    // A freshness invocation can reuse the home tab only through the worker's
    // pre-request claim. This uses the packaged MAIN/isolated/background route,
    // rather than granting a requested ID to the synthetic normalizer.
    stored.polylogueCaptureFreshnessQueue = scheduleFreshnessHint(null, {
      provider: "chatgpt", nativeId: "fixture-1", reason: "provider_revision_changed",
      nowMs: Date.now(), providerUpdatedAt: "2026-07-16T10:03:00Z",
    });
    alarmListener({ name: "polylogueCaptureFreshnessWake" });
    await vi.waitFor(() => expect(receiverPosts).toBe(3), { timeout: 15000 });
    const foregroundRequest = tabs.sendMessage.mock.calls.find(([, message]) => message.type === "polylogue.capturePage");
    expect(foregroundRequest[1].invocationRef).toEqual({ id: expect.any(String), token: expect.any(String) });
    await vi.waitFor(async () => expect(await invocationStore.getCapture(foregroundRequest[1].invocationRef.id)).toBeUndefined(), { timeout: 15000 });
    await vi.waitFor(() => expect(stored.polylogueCaptureFreshnessQueue.entries["chatgpt:fixture-1"]?.lease_owner).toBeNull(), { timeout: 15000 });
    let foregroundCache = null;
    for await (const capture of invocationStore.captures()) {
      if (capture.kind === "native-cache" && capture.owner.document_id === "packaged-document") foregroundCache = capture;
    }
    expect(foregroundCache).not.toBeNull();
    const { CaptureStaging } = await import(/* @vite-ignore */ pathToFileURL(join(unpacked, "src", "capture", "staging.js")).href);
    const foregroundRaw = await new CaptureStaging(globalThis.navigator.storage, invocationStore).metadata(foregroundCache.raw_ref.id);
    expect(foregroundRaw).toMatchObject({
      kind: "native-response", invocation_native_id: "fixture-1",
      invocation_ref: foregroundRequest[1].invocationRef,
      owner: { tab_id: 42, document_id: "packaged-document", provider: "chatgpt" },
      acquisition_sequence: expect.any(Number), observed_at: expect.any(String),
    });
    expect(pageWindow.location.pathname).toBe("/");
    expect(tabs.create).not.toHaveBeenCalled();
    expect(tabs.update).not.toHaveBeenCalled();

    replyNativeId = "wrong-native-session";
    stored.polylogueCaptureFreshnessQueue = scheduleFreshnessHint(null, {
      provider: "chatgpt", nativeId: "fixture-1", reason: "provider_revision_changed",
      nowMs: Date.now(), providerUpdatedAt: "2026-07-16T10:04:00Z",
    });
    alarmListener({ name: "polylogueCaptureFreshnessWake" });
    await vi.waitFor(() => expect(stored.polylogueCaptureFreshnessQueue.entries["chatgpt:fixture-1"]?.last_error)
      .toBe("native_capture_identity_mismatch"), { timeout: 15000 });
    expect(receiverPosts).toBe(3);
    const retainedCache = await invocationStore.getCapture(foregroundCache.id);
    expect(retainedCache.raw_ref).toEqual(foregroundCache.raw_ref);
    expect(await (await new CaptureStaging(globalThis.navigator.storage, invocationStore).file(foregroundRaw.id)).text()).toContain('"id":"fixture-1"');

    let entered;
    const enteredFetch = new Promise((resolve) => { entered = resolve; });
    let release;
    heldNativeFetch = { entered, release: new Promise((resolve) => { release = resolve; }) };
    replyNativeId = "fixture-1";
    stored.polylogueCaptureFreshnessQueue = scheduleFreshnessHint(null, {
      provider: "chatgpt", nativeId: "fixture-1", reason: "provider_revision_changed",
      nowMs: Date.now(), providerUpdatedAt: "2026-07-16T10:05:00Z",
    });
    alarmListener({ name: "polylogueCaptureFreshnessWake" });
    await enteredFetch;
    const cancelledRequest = tabs.sendMessage.mock.calls.filter(([, message]) => message.type === "polylogue.capturePage").at(-1)[1];
    let cancellationSettled = false;
    const cancellation = tabs.sendMessage(42, { type: "polylogue.cancelCapture", invocationRef: cancelledRequest.invocationRef })
      .then((result) => { cancellationSettled = true; return result; });
    await vi.waitFor(async () => expect(await invocationStore.getCapture(cancelledRequest.invocationRef.id)).toMatchObject({ state: "closing" }), { timeout: 15000 });
    expect(cancellationSettled).toBe(false);
    expect(receiverPosts).toBe(3);
    release();
    expect(await cancellation).toMatchObject({ ok: true, drained: 1 });
    heldNativeFetch = null;
    await vi.waitFor(async () => expect(await invocationStore.getCapture(cancelledRequest.invocationRef.id)).toBeUndefined(), { timeout: 15000 });
    await vi.waitFor(() => expect(stored.polylogueCaptureFreshnessQueue.entries["chatgpt:fixture-1"]?.lease_owner).toBeNull(), { timeout: 15000 });
    expect((await invocationStore.getCapture(foregroundCache.id)).raw_ref).toEqual(foregroundCache.raw_ref);

    let navigationEntered;
    const navigationFetch = new Promise((resolve) => { navigationEntered = resolve; });
    let navigationRelease;
    heldNativeFetch = { entered: navigationEntered, release: new Promise((resolve) => { navigationRelease = resolve; }) };
    stored.polylogueCaptureFreshnessQueue = scheduleFreshnessHint(null, {
      provider: "chatgpt", nativeId: "fixture-1", reason: "provider_revision_changed",
      nowMs: Date.now(), providerUpdatedAt: "2026-07-16T10:06:00Z",
    });
    alarmListener({ name: "polylogueCaptureFreshnessWake" });
    await navigationFetch;
    ownedTabs[0].url = "https://chatgpt.com/c/intervening-conversation";
    pageDom.reconfigure({ url: ownedTabs[0].url });
    navigationRelease();
    await vi.waitFor(() => expect(stored.polylogueCaptureFreshnessQueue.entries["chatgpt:fixture-1"]?.last_error)
      .toBe("native_capture_unavailable"), { timeout: 15000 });
    expect(receiverPosts).toBe(3);
    expect((await invocationStore.getCapture(foregroundCache.id)).raw_ref).toEqual(foregroundCache.raw_ref);
    ownedTabs[0].url = "https://chatgpt.com/";
    pageDom.reconfigure({ url: ownedTabs[0].url });
    heldNativeFetch = null;

    let restartEntered;
    const restartFetch = new Promise((resolve) => { restartEntered = resolve; });
    let restartRelease;
    heldNativeFetch = { entered: restartEntered, release: new Promise((resolve) => { restartRelease = resolve; }) };
    stored.polylogueCaptureFreshnessQueue = scheduleFreshnessHint(null, {
      provider: "chatgpt", nativeId: "fixture-1", reason: "provider_revision_changed",
      nowMs: Date.now(), providerUpdatedAt: "2026-07-16T10:07:00Z",
    });
    alarmListener({ name: "polylogueCaptureFreshnessWake" });
    await restartFetch;
    // Restore may legitimately pin the newer sealed native acquisition left
    // by the preceding navigation. Freeze custody after that selection, before
    // interruption of the next provider read.
    const restartCache = await invocationStore.getCapture(foregroundCache.id);
    const restartRaw = await new CaptureStaging(globalThis.navigator.storage, invocationStore).metadata(restartCache.raw_ref.id);
    const restartBytes = await (await new CaptureStaging(globalThis.navigator.storage, invocationStore).file(restartRaw.id)).text();
    const interruptedRequest = tabs.sendMessage.mock.calls.filter(([, message]) => message.type === "polylogue.capturePage").at(-1)[1];
    const providerReadsBeforeRestart = pageFetchCalls.length;
    await import(/* @vite-ignore */ `${pathToFileURL(join(unpacked, "src", "background.js")).href}?invocation-restart=${Date.now()}`);
    await vi.waitFor(async () => expect(await invocationStore.getCapture(interruptedRequest.invocationRef.id)).toMatchObject({ state: "closing" }), { timeout: 15000 });
    expect(pageFetchCalls).toHaveLength(providerReadsBeforeRestart);
    expect(receiverPosts).toBe(3);
    restartRelease();
    await vi.waitFor(async () => expect(await invocationStore.getCapture(interruptedRequest.invocationRef.id)).toBeUndefined(), { timeout: 15000 });
    await vi.waitFor(() => expect(stored.polylogueCaptureFreshnessQueue.entries["chatgpt:fixture-1"]?.lease_owner).toBeNull(), { timeout: 15000 });
    heldNativeFetch = null;
    expect((await invocationStore.getCapture(restartCache.id)).raw_ref).toEqual(restartCache.raw_ref);
    expect(await (await new CaptureStaging(globalThis.navigator.storage, invocationStore).file(restartRaw.id)).text()).toBe(restartBytes);

    const originalBegin = CaptureStaging.prototype.begin;
    const beginSpy = vi.spyOn(CaptureStaging.prototype, "begin").mockImplementation(async function (...args) {
      if (args[1]?.invocation_ref) await holdAdmission("staging");
      const result = await originalBegin.apply(this, args);
      if (args[1]?.invocation_ref) await holdAdmission("staging-reply");
      return result;
    });
    try {
      for (const [index, phase] of ["tab", "document", "staging", "staging-reply"].entries()) {
        let admissionEntered;
        const reachedAdmission = new Promise((resolve) => { admissionEntered = resolve; });
        let admissionRelease;
        admissionGate = { phase, enabled: false, entered: admissionEntered, release: new Promise((resolve) => { admissionRelease = resolve; }) };
        const readsBeforeAdmission = pageFetchCalls.length;
        stored.polylogueCaptureFreshnessQueue = scheduleFreshnessHint(null, {
          provider: "chatgpt", nativeId: "fixture-1", reason: "provider_revision_changed",
          nowMs: Date.now(), providerUpdatedAt: `2026-07-16T10:${String(10 + index).padStart(2, "0")}:00Z`,
        });
        alarmListener({ name: "polylogueCaptureFreshnessWake" });
        await reachedAdmission;
        const admissionRequest = tabs.sendMessage.mock.calls.filter(([, message]) => message.type === "polylogue.capturePage").at(-1)[1];
        const cancelledAdmission = tabs.sendMessage(42, { type: "polylogue.cancelCapture", invocationRef: admissionRequest.invocationRef });
        await vi.waitFor(async () => expect(await invocationStore.getCapture(admissionRequest.invocationRef.id)).toMatchObject({ state: "closing" }), { timeout: 15000 });
        admissionRelease();
        expect(await cancelledAdmission).toMatchObject({ ok: true });
        await vi.waitFor(async () => expect(await invocationStore.getCapture(admissionRequest.invocationRef.id)).toBeUndefined(), { timeout: 15000 });
        await vi.waitFor(() => expect(stored.polylogueCaptureFreshnessQueue.entries["chatgpt:fixture-1"]?.lease_owner).toBeNull(), { timeout: 15000 });
        expect(pageFetchCalls).toHaveLength(readsBeforeAdmission);
        expect(receiverPosts).toBe(3);
        expect((await invocationStore.getCapture(restartCache.id)).raw_ref).toEqual(restartCache.raw_ref);
        admissionGate = null;
      }
    } finally { admissionGate = null; beginSpy.mockRestore(); }

    // A fresh profile has a new storage factory and worker module graph.
    // A query on the old composition-root URL still caches its imported runtime.
    globalThis.indexedDB = new IDBFactory();
    Object.defineProperty(globalThis, "navigator", { configurable: true, value: { storage: memoryOriginStorage() } });
    const recoveryUnpacked = join(smokeRoot, "recovery-unpacked");
    execFileSync("python3", ["-c", "import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])", archive, recoveryUnpacked]);
    expect(readFileSync(join(recoveryUnpacked, "src", "background", "runtime.js")))
      .toEqual(readFileSync(join(unpacked, "src", "background", "runtime.js")));
    stored = {
      receiverBaseUrl: "http://127.0.0.1:8765",
      receiverAuthToken: "token",
      polylogueBackfillRecoveryCheckpoint: {
        version: 1,
        jobs: [{
          id: "packaged-recovered", provider: "chatgpt", cutoff: "2026-01-01T00:00:00Z", status: "running",
          inventory_cursor: "17", policy: { leaseMs: 180000, maxDailyRequests: 10 }, execution_generation: 0,
          learned_cadence_ms: 40000, daily_requests: 7, last_ack: { receiver_request_id: "ack-1", content_hash: "hash-1" },
        }],
        queue: [{ id: "packaged-recovered-item", job_id: "packaged-recovered", provider: "chatgpt", native_id: "one", state: "captured_waiting_receiver", content_hash: "hash-1" }],
        revisions: [],
      },
    };
    const pageWorkCount = pageRequests.filter((request) => request.operation !== "identity").length;
    const recoveredWorkerUrl = `${pathToFileURL(join(recoveryUnpacked, "src", "background.js")).href}?recovery=${Date.now()}`;
    await import(/* @vite-ignore */ recoveredWorkerUrl);
    const recoveredStatus = await send({ type: "polylogue.backfill.status" });
    expect(recoveredStatus).toMatchObject({ ok: true });
    // Original checkpoint input transfers to durable custody, but missing
    // acquired bytes remain an actionable refusal, never a completed capture.
    expect(recoveredStatus.jobs.find(job => job.id === "packaged-recovered")).toMatchObject({
      id: "packaged-recovered", status: "paused", cooldown_reason: "browser_profile_recovery_required",
      inventory_cursor: "17", daily_requests: 7, last_ack: { receiver_request_id: "ack-1", content_hash: "hash-1" },
      progress: expect.objectContaining({ operator_action: 1 }),
      recovery_checkpoint_error: "capture_job_account_scope_unresolved",
    });
    expect(recoveredStatus.jobs.some(job => job.id === started.job.id)).toBe(false);
    const { IndexedDbBackfillStore: RecoveredStore } = await import(/* @vite-ignore */ pathToFileURL(join(recoveryUnpacked, "src", "backfill", "storage.js")).href);
    const convertedStore = new RecoveredStore();
    expect(await convertedStore.getQueue("packaged-recovered-item")).toMatchObject({
      state: "recovery_required", last_response_class: "browser_profile_recovery_required",
    });
    expect(stored.polylogueBackfillRecoveryCheckpoint).toBeUndefined();
    expect(pageRequests.filter((request) => request.operation !== "identity")).toHaveLength(pageWorkCount);
    pageDom.window.close();
    rmSync(smokeRoot, { recursive: true, force: true });
  }, 120_000);
});
