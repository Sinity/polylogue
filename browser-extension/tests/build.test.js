// Tests for scripts/build.mjs + scripts/validate-manifest.mjs.
//
// These exercise the version sync + Firefox manifest transform + archive
// emission so a future change to the build pipeline cannot silently break
// the release artifact shape.

import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { IDBFactory, indexedDB } from "fake-indexeddb";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const __dirname = dirname(fileURLToPath(import.meta.url));
const EXT_ROOT = resolve(__dirname, "..");
const BUILD = join(EXT_ROOT, "scripts", "build.mjs");
const VALIDATE = join(EXT_ROOT, "scripts", "validate-manifest.mjs");
const MANIFEST_PATH = join(EXT_ROOT, "manifest.json");
const PACKAGE_PATH = join(EXT_ROOT, "package.json");

const ORIGINAL_MANIFEST = readFileSync(MANIFEST_PATH, "utf8");
const ORIGINAL_PACKAGE = readFileSync(PACKAGE_PATH, "utf8");
const { Headers, Response } = globalThis;

function restore() {
  writeFileSync(MANIFEST_PATH, ORIGINAL_MANIFEST);
  writeFileSync(PACKAGE_PATH, ORIGINAL_PACKAGE);
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

  it("rewrites manifest + package.json to the requested version", () => {
    execFileSync("node", [BUILD, "--version", "9.8.7", "--sync-only"], { stdio: "pipe" });
    const manifest = JSON.parse(readFileSync(MANIFEST_PATH, "utf8"));
    const pkg = JSON.parse(readFileSync(PACKAGE_PATH, "utf8"));
    expect(manifest.version).toBe("9.8.7");
    expect(pkg.version).toBe("9.8.7");
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
    let stored = { receiverBaseUrl: "http://127.0.0.1:8765", receiverAuthToken: "token" };
    let sessionStored = {};
    const pageRequests = [];
    const pageFetchCalls = [];
    const pageToken = "packaged-page-token";
    const pageAccount = "packaged-page-account";
    const pageWindow = {
      location: new URL("https://chatgpt.com/"),
      localStorage: { getItem: () => null },
      setTimeout: globalThis.setTimeout.bind(globalThis),
      clearTimeout: globalThis.clearTimeout.bind(globalThis),
      fetch: vi.fn(async (input, options = {}) => {
        const url = new URL(input);
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
        return new Response(JSON.stringify({ id: "fixture-1", mapping: { one: { message: { id: "m1", author: { role: "user" }, content: { parts: ["fixture"] } } } } }), { headers: { "Content-Type": "application/json" } });
      }),
    };
    const ownedTabs = [];
    const tabs = {
      create: vi.fn(async ({ url, active }) => {
        const tab = { id: 77, url, active, status: "complete" };
        ownedTabs.push(tab);
        return tab;
      }),
      get: vi.fn(async (tabId) => ownedTabs.find((tab) => tab.id === tabId)),
      update: vi.fn(),
      remove: vi.fn(),
      query: vi.fn(async () => []),
      sendMessage: vi.fn(async (_tabId, message) => {
        if (message.type !== "polylogue.capturePage") return undefined;
        return {
          ok: true,
          envelope: {
            polylogue_capture_kind: "browser_llm_session",
            provider_meta: { capture_fidelity: "native_full" },
            session: {
              provider: "chatgpt",
              provider_session_id: message.providerSessionId,
              turns: [{ role: "user", text: "fixture" }],
              attachments: [],
            },
          },
        };
      }),
    };
    globalThis.indexedDB = indexedDB;
    globalThis.chrome = {
      action: { setBadgeText: vi.fn(), setBadgeBackgroundColor: vi.fn() },
      alarms: { create: vi.fn(), clear: vi.fn(), onAlarm: { addListener: vi.fn((listener) => { alarmListener = listener; }) } },
      runtime: {
        id: "packaged-extension",
        getManifest: () => ({ version: "0.9.0" }),
        onInstalled: { addListener: vi.fn() },
        onStartup: { addListener: vi.fn() },
        onMessage: { addListener: vi.fn((listener) => { messageListener = listener; }) },
      },
      scripting: { executeScript: vi.fn(async (details) => {
        if (details.files) return undefined;
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
        },
        session: {
          get: vi.fn(async (defaults) => ({ ...defaults, ...sessionStored })),
          set: vi.fn(async (patch) => { sessionStored = { ...sessionStored, ...patch }; }),
          remove: vi.fn(async (key) => { delete sessionStored[key]; }),
        },
      },
      tabs: { ...tabs, onActivated: { addListener: vi.fn() }, onUpdated: { addListener: vi.fn() } },
    };
    const fetchCalls = [];
    let receiverPosts = 0;
    globalThis.fetch = vi.fn(async (url, options = {}) => {
      fetchCalls.push({ url, options });
      let body;
      if (String(url).endsWith("/v1/browser-captures/capabilities")) {
        return { ok: true, status: 200, headers: { get: (name) => name === "X-Request-ID" ? "packaged-capability" : null }, json: async () => ({ durable_ack_fields: ["receiver_request_id", "content_hash"] }) };
      }
      if (String(url).includes("/v1/backfill-checkpoint")) {
        // Best-effort ledger-checkpoint mirror/restore traffic (polylogue-06zm)
        // is orthogonal to this fixture's receiverPosts accounting below --
        // it must neither increment that counter nor be hashed as a capture
        // envelope body.
        if (options.method === "POST") {
          return { ok: true, status: 202, headers: { get: () => null }, json: async () => ({ ok: true, extension_instance_id: "packaged-instance", stored_at: new Date().toISOString(), bytes_written: 1 }) };
        }
        return { ok: false, status: 404, headers: { get: () => null }, json: async () => ({ ok: false, error: "checkpoint_not_found" }) };
      }
      const receiverPath = new URL(url).pathname;
      if (receiverPath === "/v1/capture-jobs/discover") {
        return { ok: true, status: 200, headers: { get: () => null }, json: async () => ({ jobs: [] }) };
      }
      if (receiverPath === "/v1/capture-jobs") {
        return { ok: true, status: 201, headers: { get: () => null }, json: async () => ({ job: {
          job_id: "packaged-capture-job", provider: "chatgpt", revision: 0, lease_generation: 0,
        } }) };
      }
      if (receiverPath.endsWith("/adopt")) {
        return { ok: true, status: 200, headers: { get: () => null }, json: async () => ({
          job: { job_id: "packaged-capture-job", provider: "chatgpt", revision: 1, lease_generation: 1 },
          lease: { lease_id: "packaged-lease", generation: 1, proof: "packaged-proof" },
        }) };
      }
      if (receiverPath.endsWith("/update")) {
        return { ok: true, status: 200, headers: { get: () => null }, json: async () => ({
          job: {
            job_id: "packaged-capture-job", provider: "chatgpt", revision: 2, lease_generation: 1,
            lease_expires_at: "2026-07-16T10:02:00Z",
          },
          receipt: { kind: "capture_job_update" },
        }) };
      }
      if (receiverPath.endsWith("/checkpoint")) {
        return { ok: true, status: 200, headers: { get: () => null }, json: async () => ({
          job: { job_id: "packaged-capture-job", provider: "chatgpt", revision: 3 }, receipt: {},
        }) };
      }
      const contentHash = createHash("sha256").update(options.body, "utf8").digest("hex");
      receiverPosts += 1;
      body = receiverPosts === 1
        ? { ok: true, provider: "chatgpt", provider_session_id: "fixture-1", content_hash: contentHash }
        : { ok: true, provider: "chatgpt", provider_session_id: "fixture-1", content_hash: contentHash };
      return { ok: true, status: 200, headers: { get: (name) => name === "X-Request-ID" && receiverPosts > 1 ? "packaged-ack" : null }, json: async () => body };
    });
    const packagedWorkerUrl = `${pathToFileURL(join(unpacked, "src", "background.js")).href}?smoke=${Date.now()}`;
    await import(/* @vite-ignore */ packagedWorkerUrl);
    const send = (message) => new Promise((resolve) => messageListener(message, {}, resolve));
    const started = await send({ type: "polylogue.backfill.start", provider: "chatgpt", cutoff: "2026-01-01T00:00:00Z", policy: { baseCadenceMs: 0 } });
    await vi.waitFor(() => expect(pageRequests.some((message) => message.operation === "inventory")).toBe(true));
    for (let inventoryCount = 2; inventoryCount <= 4; inventoryCount += 1) {
      alarmListener({ name: `polylogueBackfillWake:${started.job.id}` });
      await vi.waitFor(() => expect(pageRequests.filter((message) => message.operation === "inventory")).toHaveLength(inventoryCount));
    }
    alarmListener({ name: `polylogueBackfillWake:${started.job.id}` });
    await vi.waitFor(() => expect(receiverPosts).toBe(1));
    const paused = await send({ type: "polylogue.backfill.status" });
    expect(paused.jobs[0]).toMatchObject({
      status: "paused",
      cooldown_reason: "receiver_contract_incompatible",
      last_error: "receiver_contract_incompatible:missing_receiver_request_id",
    });
    expect(receiverPosts).toBe(1);
    await send({ type: "polylogue.backfill.control", job_id: started.job.id, action: "resume" });
    alarmListener({ name: `polylogueBackfillWake:${started.job.id}` });
    await vi.waitFor(() => expect(receiverPosts).toBe(2));
    const recovered = await send({ type: "polylogue.backfill.status" });
    expect(recovered.jobs[0].progress.complete).toBe(1);
    expect(pageRequests.filter((message) => message.operation === "conversation")).toHaveLength(1);
    expect(tabs.create).toHaveBeenCalledWith({ url: "https://chatgpt.com/", active: false });
    expect(tabs.update).not.toHaveBeenCalled();
    expect(pageRequests.filter((message) => message.operation !== "identity").map((message) => message.operation))
      .toEqual(["inventory", "inventory", "inventory", "inventory", "conversation"]);
    expect(pageFetchCalls.filter((call) => call.url.pathname === "/api/auth/session").length).toBeGreaterThanOrEqual(5);
    expect(JSON.stringify(pageRequests)).not.toContain(pageToken);
    expect(JSON.stringify(pageRequests)).not.toContain(pageAccount);
    expect(tabs.sendMessage).toHaveBeenCalledWith(77, expect.objectContaining({
      type: "polylogue.capturePage",
      reason: "backfill_exact_capture",
      providerSessionId: "fixture-1",
      deferReceiver: true,
      nativePayload: expect.objectContaining({ id: "fixture-1" }),
    }));
    expect(fetchCalls.every((call) => String(call.url).includes("127.0.0.1"))).toBe(true);

    globalThis.indexedDB = new IDBFactory();
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
    const recoveredWorkerUrl = `${pathToFileURL(join(unpacked, "src", "background.js")).href}?recovery=${Date.now()}`;
    await import(/* @vite-ignore */ recoveredWorkerUrl);
    const recoveredStatus = await send({ type: "polylogue.backfill.status" });
    expect(recoveredStatus.jobs[0]).toMatchObject({
      id: "packaged-recovered", status: "paused", cooldown_reason: "browser_profile_recovery_required",
      progress: { operator_action: 1 },
    });
    alarmListener({ name: "polylogueBackfillWake:packaged-recovered" });
    await Promise.resolve();
    expect(pageRequests.filter((request) => request.operation !== "identity")).toHaveLength(pageWorkCount);
    rmSync(smokeRoot, { recursive: true, force: true });
  }, 15_000);
});
