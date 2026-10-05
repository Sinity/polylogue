import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { createContext, Script } from "node:vm";
import { webcrypto } from "node:crypto";
import { describe, expect, it, vi } from "vitest";

const source = name => readFileSync(resolve(dirname(fileURLToPath(import.meta.url)), `../../src/content/${name}.js`), "utf8");
const ownerA = "a".repeat(32), ownerB = "b".repeat(32), ownerC = "c".repeat(32);
const flush = () => new Promise(resolve => globalThis.setImmediate(resolve));

// Distinct MAIN and ISOLATED globals share only page messages, like Chrome.
// Runtime calls remain installation-local. No page message is translated.
function page({ mainRuntime, provider = "claude", fetch = async () => new globalThis.Response("{}", { headers: { "content-type": "application/json" } }) } = {}) {
  const realms = [], posted = [], errors = [];
  const location = new URL(provider === "claude" ? "https://claude.ai/chat/neutral" : "https://chatgpt.com/c/neutral");
  function realm(ownerId, { eligible = false, refuse = false } = {}) {
    const listeners = new Map(), calls = [], storageListeners = [];
    const stored = { polylogueAmbientSettings: { enabled: eligible, automatic_capture_enabled: eligible },
      polylogueReceiverPairing: refuse ? null : { receiver_id: "neutral", state: "online" } };
    const window = { location, crypto: webcrypto, btoa: value => globalThis.Buffer.from(value, "binary").toString("base64"), fetch,
      document: { getElementById: () => null }, performance: { getEntriesByType: () => [] },
      localStorage: { getItem: () => JSON.stringify({ orgUuid: "11111111-1111-4111-8111-111111111111" }) },
      addEventListener(type, listener) { if (!listeners.has(type)) listeners.set(type, new Set()); listeners.get(type).add(listener); },
      removeEventListener(type, listener) { listeners.get(type)?.delete(listener); },
      postMessage(data) {
        posted.push(data);
        globalThis.queueMicrotask(() => { for (const target of realms) for (const fn of target.listeners.get("message") || []) {
          try { Promise.resolve(fn({ source: target.window, origin: location.origin, data })).catch(error => errors.push(error)); }
          catch (error) { errors.push(error); }
        } });
      } };
    const chrome = ownerId ? { runtime: { id: ownerId, onMessage: { addListener() {} }, async sendMessage(message) {
      calls.push(message);
      if (message.type === "polylogue.asset.begin") return refuse ? { ok: false, error: "receiver_unpaired" } : { ok: true, ref: { id: `${ownerId}:${message.request_id}` } };
      if (message.type === "polylogue.asset.seal") return { ok: true, asset: { staged_asset: message.ref } };
      return { ok: true };
    } }, storage: { local: { async get() { return stored; } }, onChanged: { addListener(fn) { storageListeners.push(fn); } } } } : (mainRuntime ? { runtime: mainRuntime } : undefined);
    const context = createContext({ window, chrome, crypto: webcrypto, AbortController: globalThis.AbortController, DOMException: globalThis.DOMException, URL, Response: globalThis.Response, Uint8Array, TextEncoder, setTimeout: globalThis.setTimeout, clearTimeout: globalThis.clearTimeout });
    const result = { window, listeners, calls, stored, storageListeners, context,
      load(name) { new Script(source(name)).runInContext(context); },
      async configure(settings, pairing = stored.polylogueReceiverPairing) { stored.polylogueAmbientSettings = settings; stored.polylogueReceiverPairing = pairing; storageListeners.forEach(fn => fn({ polylogueAmbientSettings: { newValue: settings } }, "local")); await flush(); },
    };
    realms.push(result); return result;
  }
  const main = realm(null);
  return { main, realm, posted, errors };
}

async function explicit(owner, requestId = "neutral-request") {
  const result = new Promise(resolve => owner.listeners.get("message").add(event => {
    const data = owner.window.polylogueAssetStream.readPageMessage(event);
    if (data?.type === "polylogue.claude.nativeFetchResponse" && data.requestId === requestId) resolve(data);
  }));
  owner.window.polylogueAssetStream.pageMessage({ type: "polylogue.claude.nativeFetchRequest", requestId, conversationId: "neutral" });
  return result;
}

describe("installation-owned original page transport", () => {
  it.each([undefined, null, "", "foreign", 123])("uses MAIN scoped frames when external runtime API has no valid owner (%s)", async id => {
    const external = vi.fn(() => { throw new Error("external runtime used"); });
    const h = page({ mainRuntime: { id, sendMessage: external } }), candidate = h.realm(ownerA);
    h.main.load("asset_stream"); candidate.load("asset_stream"); h.main.load("claude_bridge");
    const result = await explicit(candidate);
    expect(result.capture.ok).toBe(true);
    expect(result.ownerId).toBe(ownerA);
    expect(candidate.calls.filter(call => call.type === "polylogue.asset.begin")).toHaveLength(1);
    expect(external).not.toHaveBeenCalled();
    expect(h.posted.filter(frame => frame.type.endsWith(".claude.nativeFetchResponse"))).toHaveLength(1);
    expect(h.errors).toEqual([]);
  });

  it.each([ownerA, ownerB])("keeps true ISOLATED decoding bound to its actual owner (%s)", id => {
    const h = page(), candidate = h.realm(id);
    candidate.load("asset_stream");
    const frame = owner => ({ source: candidate.window, origin: candidate.window.location.origin,
      data: { type: `polylogue.page.v2.${owner}.claude.nativeFetchRequest`, requestId: "neutral" } });
    expect(candidate.window.polylogueAssetStream.readPageMessage(frame(id)).ownerId).toBe(id);
    expect(candidate.window.polylogueAssetStream.readPageMessage(frame(id === ownerA ? ownerB : ownerA))).toBeNull();
    expect(candidate.window.polylogueAssetStream.readPageMessage(frame("foreign"))).toBeNull();
  });

  it("ignores foreign-first and legacy responders while replacing their stale singleton claims", async () => {
    const h = page(), foreign = h.realm(ownerB, { refuse: true }), candidate = h.realm(ownerA);
    h.main.window.polylogueAssetStream = { prepareResponse() { throw new Error("stale helper called"); } };
    h.main.window.__polylogueClaudeFetchHookInstalled = true;
    // Previously installed responder sees the same request first and injects
    // both an unowned legacy reply and a foreign-owner reply with its nonce.
    foreign.window.addEventListener("message", event => {
      if (!event.data.type.endsWith("responseStart")) return;
      const requestId = event.data.requestId;
      foreign.window.postMessage({ type: "polylogue.responseReady", requestId, ok: false, error: "receiver_unpaired" });
      foreign.window.postMessage({ type: `polylogue.page.v2.${ownerB}.responseReady`, requestId, ok: false, error: "receiver_unpaired" });
    });
    h.main.load("asset_stream"); foreign.load("asset_stream"); candidate.load("asset_stream"); h.main.load("claude_bridge");
    const result = await explicit(candidate);
    expect(result.capture.ok).toBe(true);
    expect(result.capture.bodyRef.id.startsWith(ownerA)).toBe(true);
    expect(candidate.calls.filter(call => call.type === "polylogue.asset.begin")).toHaveLength(1);
    expect(foreign.calls).toEqual([]);
    expect(h.errors).toEqual([]);
  });

  it("keeps identical request nonces independently owned and cancellation cannot abort a foreign acquisition", async () => {
    const releases = [], fetch = vi.fn((_url, options) => new Promise((resolve, reject) => {
      releases.push(() => resolve(new globalThis.Response("{}", { headers: { "content-type": "application/json" } })));
      options.signal.addEventListener("abort", () => reject(options.signal.reason), { once: true });
    }));
    const h = page({ fetch }), a = h.realm(ownerA), b = h.realm(ownerB);
    h.main.load("asset_stream"); a.load("asset_stream"); b.load("asset_stream"); h.main.load("claude_bridge");
    const first = explicit(a, "same"), second = explicit(b, "same");
    await vi.waitFor(() => expect(releases).toHaveLength(2));
    a.window.polylogueAssetStream.pageMessage({ type: "polylogue.claude.cancelRequest", requestId: "same" });
    expect((await first).error).toBe("capture_cancelled");
    releases[1](); expect((await second).capture.ok).toBe(true);
    expect(a.calls.some(call => call.type === "polylogue.asset.discard")).toBe(true);
    expect(b.calls.some(call => call.type === "polylogue.asset.seal")).toBe(true);
    expect(h.errors).toEqual([]);
  });

  it("observes each eligible owner with its own clone and preserves the original app response across stale wrappers", async () => {
    const original = new globalThis.Response('{"id":"neutral"}', { headers: { "content-type": "application/json" } });
    const fetch = vi.fn(async () => original), h = page({ provider: "chatgpt", fetch });
    const a = h.realm(ownerA, { eligible: true }), b = h.realm(ownerB, { eligible: true }), c = h.realm(ownerC, { eligible: true, refuse: true });
    h.main.window.__polylogueFetchHookInstalled = true;
    h.main.window.polylogueAssetStream = { stageResponse: vi.fn() };
    // Actual old hook pattern: a borrowed clone is handed to the global
    // helper without an owner. New helper refuses it before admission.
    const originalFetch = h.main.window.fetch;
    h.main.window.fetch = async (...args) => { const response = await originalFetch(...args);
      void h.main.window.polylogueAssetStream.stageResponse(response.clone(), "chatgpt", new globalThis.AbortController().signal).catch(() => undefined); return response; };
    h.main.load("asset_stream"); a.load("asset_stream"); b.load("asset_stream"); c.load("asset_stream"); h.main.load("chatgpt_bridge");
    await flush();
    await b.configure({ enabled: true, automatic_capture_enabled: true }, { receiver_id: "neutral", state: "offline" });
    expect(h.main.window.polylogueAssetStream.eligibleOwners()).toEqual([ownerA, ownerB]);
    const response = await h.main.window.fetch("https://chatgpt.com/backend-api/conversation/neutral");
    expect(response).toBe(original); expect(await response.text()).toBe('{"id":"neutral"}');
    await vi.waitFor(() => expect([a, b].map(owner => owner.calls.filter(call => call.type === "polylogue.asset.seal").length)).toEqual([1, 1]));
    expect(c.calls).toEqual([]); expect(fetch).toHaveBeenCalledTimes(1);
    await b.configure({ enabled: false, automatic_capture_enabled: false });
    expect(h.main.window.polylogueAssetStream.eligibleOwners()).toEqual([ownerA]);
    expect(h.errors).toEqual([]);
  });
  it("keeps same-nonce chunks and terminal seals under each original staging owner", async () => {
    const h = page(), a = h.realm(ownerA), b = h.realm(ownerB);
    h.main.load("asset_stream"); a.load("asset_stream"); b.load("asset_stream");
    const acquire = (owner, text) => owner.window.polylogueAssetStream.request({ provider: "chatgpt", requestId: "same-stream", signal: new globalThis.AbortController().signal,
      start: async () => ({ status: "acquired", asset: await h.main.window.polylogueAssetStream.stream(new globalThis.Response(text), "same-stream", new globalThis.AbortController().signal, { ownerId: owner === a ? ownerA : ownerB }) }) });
    const results = await Promise.all([acquire(a, "alpha"), acquire(b, "beta")]);
    expect(results.map(result => result.asset.staged_asset.id)).toEqual([`${ownerA}:same-stream`, `${ownerB}:same-stream`]);
    for (const [owner, text, id] of [[a, "alpha", ownerA], [b, "beta", ownerB]]) {
      const chunks = owner.calls.filter(call => call.type === "polylogue.asset.chunk");
      expect(chunks).toHaveLength(1);
      expect(chunks[0].ref.id).toBe(`${id}:same-stream`);
      expect(globalThis.Buffer.from(chunks[0].base64, "base64").toString()).toBe(text);
      expect(owner.calls.filter(call => call.type === "polylogue.asset.seal")).toHaveLength(1);
    }
    expect(h.errors).toEqual([]);
  });

  it("revokes passive registration at page cleanup and never admits an ownerless stale clone", async () => {
    const h = page({ provider: "chatgpt" }), a = h.realm(ownerA, { eligible: true });
    h.main.load("asset_stream"); a.load("asset_stream"); await flush();
    expect(h.main.window.polylogueAssetStream.eligibleOwners()).toEqual([ownerA]);
    const original = new globalThis.Response("unread original");
    await expect(h.main.window.polylogueAssetStream.stageResponse(original.clone(), "chatgpt", new globalThis.AbortController().signal)).rejects.toThrow("capture_transport_owner_unavailable");
    expect(a.calls).toEqual([]); expect(await original.text()).toBe("unread original");
    for (const target of [h.main, a]) for (const fn of target.listeners.get("pagehide") || []) fn();
    expect(h.main.window.polylogueAssetStream.eligibleOwners()).toEqual([]);
    expect(h.errors).toEqual([]);
  });

});
