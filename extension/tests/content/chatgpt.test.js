import { stagingRuntime } from "../infra/capture-staging.js";
/**
 * Tests for chatgpt.js's URL parsing, on-demand native fetch, and asset
 * descriptor identification, driven through the REAL source (common.js +
 * chatgpt_bridge.js + chatgpt.js loaded via vm.Script into a JSDOM window,
 * the same technique tests/content/chatgpt_bridge.test.js and
 * tests/content/grok.test.js already use) rather than hand-copied function
 * bodies.
 *
 * This file used to keep local copies of conversationIdFromUrl,
 * fetchNativePayloadOnDemand, collectAssetDescriptors, and
 * sandboxPathsFromText that had to be manually kept in sync with the
 * source. That divergence risk is exactly how src/common.js's buildEnvelope
 * silently dropping every turn's `blocks` field (polylogue-ah21 regressed)
 * went unnoticed for as long as it did in the sibling content-script test
 * files -- a copy tests itself, not the production code. All coverage here
 * now exercises window.polylogueCapture.capturePage (the one function the
 * content script IIFE actually exposes) against the real IIFE bodies.
 */

import { Buffer } from "node:buffer";
import { webcrypto } from "node:crypto";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { Script } from "node:vm";

import { JSDOM } from "jsdom";
import { afterEach, describe, expect, it, vi } from "vitest";


const testDirectory = dirname(fileURLToPath(import.meta.url));
const assetStreamSource = readFileSync(resolve(testDirectory, "../../src/content/asset_stream.js"), "utf8");
const bridgeSource = readFileSync(resolve(testDirectory, "../../src/content/chatgpt_bridge.js"), "utf8");
const commonSource = readFileSync(resolve(testDirectory, "../../src/common.js"), "utf8");
const contentSource = readFileSync(resolve(testDirectory, "../../src/content/chatgpt.js"), "utf8");
const openDoms = [];

function jsonResponse(body, status = 200) {
  return new globalThis.Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
}

function notFoundResponse() {
  return new globalThis.Response(JSON.stringify({ detail: "not_found" }), { status: 404, headers: { "content-type": "application/json" } });
}

// Requests not declared by the fixture return404, preserving descriptor and
// refusal behavior through the actual asset acquisition route.
function installChatgpt({ url = "https://chatgpt.com/c/conversation-1", fetch, runtimeMessage = null, storageFailure = false } = {}) {
  const dom = new JSDOM("<!doctype html><title>ChatGPT fixture</title>", { url, runScripts: "outside-only" });
  openDoms.push(dom);
  const cryptoAdapter = {
    randomUUID: () => webcrypto.randomUUID(),
    subtle: {
      digest(algorithm, data) {
        return webcrypto.subtle.digest(algorithm, Buffer.from(new dom.window.Uint8Array(data)));
      },
    },
  };
  Object.defineProperty(dom.window, "crypto", { configurable: true, value: cryptoAdapter });
  Object.defineProperty(dom.window, "fetch", { configurable: true, writable: true, value: fetch || (async () => notFoundResponse()) });
  const runtimeListeners = [];
  const storageListeners = new Set();
  const nativeCaptures = [];
  const chrome = {
    storage: { local: { get: async () => ({ polylogueAmbientSettings: {}, polylogueReceiverPairing: { receiver_id: "neutral", state: "online" } }) }, onChanged: { addListener: listener => { if (storageFailure) throw new Error("synthetic diagnostic observer refusal"); storageListeners.add(listener); }, removeListener: listener => storageListeners.delete(listener) } },
    runtime: {
      id: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      getManifest: () => ({ version: "0.1.0" }),
      onMessage: { addListener: (listener) => runtimeListeners.push(listener) },
      async sendMessage(message) {
        const assetResult = await dom.__captureRuntime.sendMessage(message);
        const overridden = await runtimeMessage?.(message);
        // Observe the staging owner's validated rate response as well as the
        // page messages, while keeping its reply authoritative.
        if (assetResult !== undefined) return assetResult;
        if (overridden !== undefined) return overridden;
        if (message.type === "polylogue.capture") {
          return { ok: true, provider: "chatgpt", provider_session_id: "conversation-1", receiver_request_id: "synthetic-request" };
        }
        if (message.type === "polylogue.archiveState") return { captured: true, state: "archived" };
        return { ok: true };
      },
    },
  };
  dom.__captureRuntime = stagingRuntime();
  Object.defineProperty(dom.window, "chrome", { configurable: true, value: chrome });
  Object.defineProperty(dom.window, "postMessage", {
    configurable: true,
    value(data) {
      dom.window.queueMicrotask(() => {
        if (!dom.window.document) return;
        if (data.type === "polylogue.page.v2.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.chatgpt.nativeCapture") nativeCaptures.push(data.capture);
        dom.window.dispatchEvent(new dom.window.MessageEvent("message", { source: dom.window, origin: dom.window.location.origin, data }));
      });
    },
  });
  const context = dom.getInternalVMContext();
  new Script(assetStreamSource).runInContext(context);
  new Script(bridgeSource).runInContext(context);
  new Script(commonSource).runInContext(context);
  new Script(contentSource).runInContext(context);
  function sendRuntimeMessage(message) {
    return new Promise((resolve, reject) => {
      const listener = runtimeListeners.find((candidate) => candidate(message, {}, resolve) === true);
      if (!listener) reject(new Error(`no runtime listener accepted ${message.type}`));
    });
  }
  dom.__captureRuntime.setDispatch(sendRuntimeMessage);
  async function sendCaptureMessage(message) {
    const result = await sendRuntimeMessage(message);
    if (result?.envelope) result.envelope = await dom.__captureRuntime.materialize(result.envelope);
    return result;
  }
  return { dom, nativeCaptures, sendRuntimeMessage: sendCaptureMessage, storageListeners, emitStorage: (changes, area = "local") => { for (const listener of storageListeners) listener(changes, area); } };
}

afterEach(() => {
  for (const dom of openDoms.splice(0)) {
    dom.window.dispatchEvent(new dom.window.Event("pagehide"));
    dom.window.close();
  }
});

describe("connected capture recovery", () => {
  it("suspends queued freshness work on pagehide and resumes observation on pageshow", async () => {
    vi.useFakeTimers();
    try {
      const messages = [];
      const { dom } = installChatgpt({ runtimeMessage: (message) => { messages.push(message); } });
      const article = dom.window.document.createElement("article");
      article.textContent = "first visible text";
      dom.window.document.body.append(article);
      await Promise.resolve();
      dom.window.dispatchEvent(new dom.window.Event("pagehide"));
      await vi.advanceTimersByTimeAsync(2000);
      expect(messages.filter((message) => message.type === "polylogue.captureFreshnessHint")).toEqual([]);
      dom.window.dispatchEvent(new dom.window.Event("pageshow"));
      await vi.advanceTimersByTimeAsync(2000);
      expect(messages.filter((message) => message.type === "polylogue.captureFreshnessHint"))
        .toEqual([expect.objectContaining({ provider: "chatgpt", provider_session_id: "conversation-1" })]);
      dom.window.dispatchEvent(new dom.window.Event("pagehide"));
    } finally { vi.useRealTimers(); }
  });

  it("cancels and drains concurrent record asset acquisitions sharing a raw revision", async () => {
    const { dom, sendRuntimeMessage } = installChatgpt();
    let started; const ready = new Promise((resolve) => { started = resolve; });
    let requests = 0; let cancelled = 0;
    dom.window.polylogueAssetStream.request = ({ signal }) => new Promise((_resolve, reject) => {
      signal.addEventListener("abort", () => { cancelled += 1; reject(signal.reason); }, { once: true });
      if (++requests === 2) started();
    });
    const request = { type: "polylogue.acquireRecordAssets", provider: "chatgpt", nativeId: "session", capture_ref: "same-raw-revision", recordKey: "first", attachmentOrdinal: 0, attachments: [{ provider_attachment_id: "asset", name: "asset.txt", provider_meta: { provider_file_id: "asset" } }] };
    const first = sendRuntimeMessage(request);
    const second = sendRuntimeMessage({ ...request, recordKey: "second" });
    await ready;
    expect(await sendRuntimeMessage({ type: "polylogue.cancelRecordAssets", capture_ref: request.capture_ref }))
      .toMatchObject({ ok: true, outcome: "cancelled" });
    expect(cancelled).toBe(2);
    expect(await first).toMatchObject({ ok: false, error: "capture_cancelled" });
    expect(await second).toMatchObject({ ok: false, error: "capture_cancelled" });
  });

  it("cancels and drains an owned native fetch without accepting a capture", async () => {
    let started = false;
    let aborted = false;
    const captures = [];
    const fetch = vi.fn((_url, options = {}) => new Promise((_resolve, reject) => {
      started = true;
      options.signal.addEventListener("abort", () => {
        aborted = true;
        reject(new globalThis.DOMException("cancelled", "AbortError"));
      }, { once: true });
    }));
    const { sendRuntimeMessage } = installChatgpt({ fetch, runtimeMessage: (message) => {
      if (message.type === "polylogue.capture") captures.push(message);
    } });
    const pending = sendRuntimeMessage({ type: "polylogue.capturePage" });
    await vi.waitFor(() => expect(started).toBe(true));
    expect(await sendRuntimeMessage({ type: "polylogue.cancelCapture" })).toMatchObject({ ok: true, outcome: "cancelled", drained: 1 });
    expect(await pending).toMatchObject({ ok: false, outcome: "cancelled" });
    expect(aborted).toBe(true);
    expect(captures).toEqual([]);
  });

  it.each([false, true])("retains thoughts with ordinary text present=%s through actual native acquisition", async (mixed) => {
    const native = { id: "conversation-1", mapping: { reasoning: { message: {
      id: "reasoning-1", author: { role: "assistant" },
      content: { content_type: mixed ? "text" : "thoughts", parts: mixed ? ["ordinary text"] : [], thoughts: [{ content: "reasoning text" }] },
    } } } };
    const fetch = vi.fn(async (input) => {
      const url = new URL(String(input), "https://chatgpt.com");
      return url.pathname === "/backend-api/conversation/conversation-1" ? new globalThis.Response(JSON.stringify(native), { headers: { "content-type": "application/json" } }) : notFoundResponse();
    });
    const { dom, sendRuntimeMessage } = installChatgpt({ fetch });
    const result = await sendRuntimeMessage({ type: "polylogue.capturePage", deferReceiver: true, providerSessionId: "conversation-1" });
    expect(result.ok).toBe(true);
    expect(await dom.__captureRuntime.retainedNativeReplies(result.envelope)).toEqual(native);
    expect(result.envelope.receiver_native.sha256).toBe("b".repeat(64));
    // Canonical thinking/text blocks: test_chatgpt_thoughts_node_produces_nonempty_thinking_text_end_to_end.
    expect(result.envelope.session.turns).toEqual([]);
  });

  it("observes start, progress and completion during continual transcript mutations", async () => {
    const hints = [];
    const { dom } = installChatgpt({ runtimeMessage: (message) => {
      if (message.type === "polylogue.captureFreshnessHint") hints.push(message);
    } });
    let now = 100_000;
    vi.spyOn(dom.window.Date, "now").mockImplementation(() => now);
    const stop = dom.window.document.createElement("button");
    stop.setAttribute("data-testid", "stop-button");
    dom.window.document.body.append(stop);
    const transcript = dom.window.document.createElement("div");
    dom.window.document.body.append(transcript);
    let mutations = 0;
    const timer = dom.window.setInterval(() => { transcript.textContent = `stream-${++mutations}`; }, 20);
    const observed = (state) => hints.some((hint) => hint.generation_observations?.some((item) => item.state === state));
    try {
      await vi.waitFor(() => expect(observed("started")).toBe(true), { timeout: 3000 });
      expect(mutations).toBeGreaterThan(10);
      expect(stop.isConnected).toBe(true);
      now += 31_000;
      await vi.waitFor(() => expect(observed("in_progress")).toBe(true), { timeout: 3000 });
      stop.remove();
      await vi.waitFor(() => expect(observed("completed")).toBe(true));
      expect(mutations).toBeGreaterThan(20);
    } finally {
      dom.window.clearInterval(timer);
    }
  });
});

describe("chatgpt.js on-demand native fetch, exact-provider capture", () => {
  it.each([
    [[{ content: "reasoning text", summary: "short summary" }], "reasoning text"],
    [[{ summary: "summary only" }, { content: "next thought" }], "summary only\nnext thought"],
  ])("preserves thoughts-only content through exact capture %#", async (thoughts, text) => {
    const payload = {
      id: "conversation-1", mapping: {
        reasoning: { parent: null, message: { id: "reasoning-1", author: { role: "assistant" }, content: { content_type: "thoughts", thoughts } } },
      },
    };
    const fetch = vi.fn(async () => jsonResponse(payload));
    const { dom, sendRuntimeMessage } = installChatgpt({ fetch });
    const result = await sendRuntimeMessage({ type: "polylogue.capturePage", reason: "backfill_exact_capture", providerSessionId: "conversation-1", deferReceiver: true });
    expect(result).toMatchObject({ ok: true, deferred: true, envelope: { session: { provider_session_id: "conversation-1", turns: [] }, receiver_native: expect.any(Object) } });
    const retained = await dom.__captureRuntime.retainedNativeReplies(result.envelope);
    expect(retained).toEqual(payload);
    expect(retained.mapping.reasoning.message.content.thoughts.map(item => item.content || item.summary).join("\n")).toBe(text);
    expect(fetch.mock.calls.map(([url]) => new globalThis.URL(url).pathname)).toEqual(["/api/auth/session", "/backend-api/conversation/conversation-1"]);
  });

  it("reads a temporary capture identity from the original document without another provider fetch", async () => {
    const url = "https://chatgpt.com/?temporary-chat=true";
    const fetch = vi.fn(async () => jsonResponse({ id: "temp-1", is_temporary: true, mapping: {} }));
    const { dom, nativeCaptures, sendRuntimeMessage } = installChatgpt({ url, fetch });
    expect(await sendRuntimeMessage({ type: "polylogue.captureIdentity", expectedUrl: url })).toEqual({ provider_session_id: null });
    await vi.waitFor(() => expect(dom.window.polylogueAssetStream.eligibleOwners()).toEqual([dom.window.chrome.runtime.id]));
    await dom.window.fetch("/backend-api/conversation/temp-1");
    await vi.waitFor(() => expect(nativeCaptures).toMatchObject([{ ok: true, bodyRef: expect.any(Object) }]));
    expect(JSON.parse(await (await dom.__captureRuntime.staging.file(nativeCaptures[0].bodyRef.id)).text()).id).toBe("temp-1");
    await vi.waitFor(async () => expect(await sendRuntimeMessage({ type: "polylogue.captureIdentity", expectedUrl: url })).toEqual({ provider_session_id: "temp-1" }));
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(await sendRuntimeMessage({ type: "polylogue.captureIdentity", expectedUrl: "https://chatgpt.com/c/other" })).toEqual({ provider_session_id: null });
    dom.reconfigure({ url: "https://chatgpt.com/c/other" });
    expect(await sendRuntimeMessage({ type: "polylogue.captureIdentity", expectedUrl: "https://chatgpt.com/c/other" })).toEqual({ provider_session_id: null });
  });

  it.each([
    { id: "ordinary-1", is_temporary: false, mapping: {} },
    { id: "bad/id", is_temporary: true, mapping: {} },
    { id: "missing-mapping", is_temporary: true },
  ])("refuses unrelated or malformed temporary capture identity %#", async (payload) => {
    const url = "https://chatgpt.com/?temporary-chat=true";
    const { dom, nativeCaptures, sendRuntimeMessage } = installChatgpt({ url, fetch: async () => jsonResponse(payload) });
    await vi.waitFor(() => expect(dom.window.polylogueAssetStream.eligibleOwners()).toEqual([dom.window.chrome.runtime.id]));
    await dom.window.fetch("/backend-api/conversation/neutral-id");
    await vi.waitFor(() => expect(nativeCaptures).toHaveLength(1));
    await new Promise((resolve) => dom.window.setTimeout(resolve, 0));
    expect(await sendRuntimeMessage({ type: "polylogue.captureIdentity", expectedUrl: url })).toEqual({ provider_session_id: null });
  });

  it("fetches the current conversation JSON with credentials for an exact capture", async () => {
    const calls = [];
    const fetch = vi.fn(async (input, options = {}) => {
      const url = new URL(String(input), "https://chatgpt.com");
      calls.push({ pathname: url.pathname, credentials: options.credentials, cache: options.cache });
      if (url.pathname === "/backend-api/conversation/conv-123") {
        return jsonResponse({ conversation_id: "conv-123", title: "Native ChatGPT title", mapping: { node: { id: "node", parent: null, message: { id: "m", author: { role: "user" }, content: { parts: ["hello"] } } } } });
      }
      return notFoundResponse();
    });
    const { sendRuntimeMessage } = installChatgpt({ url: "https://chatgpt.com/c/conv-123", fetch });

    const result = await sendRuntimeMessage({ type: "polylogue.capturePage", reason: "completion_monitor", providerSessionId: "conv-123" });

    expect(result).toMatchObject({ ok: true, envelope: { session: { provider_session_id: "conv-123", turns: [] } } });
    expect(calls.find((call) => call.pathname === "/backend-api/conversation/conv-123")).toMatchObject({ credentials: "include", cache: "no-store" });
  });

  it("supports custom GPT conversation routes", async () => {
    const fetch = vi.fn(async (input) => {
      const url = new URL(String(input), "https://chatgpt.com");
      if (url.pathname === "/backend-api/conversation/conv-123") {
        return jsonResponse({ id: "conv-123", mapping: { node: { id: "node", parent: null, message: { id: "m", author: { role: "user" }, content: { parts: ["hi"] } } } } });
      }
      return notFoundResponse();
    });
    const { sendRuntimeMessage } = installChatgpt({ url: "https://chatgpt.com/g/g-p-abc/c/conv-123", fetch });

    const result = await sendRuntimeMessage({ type: "polylogue.capturePage", reason: "completion_monitor", providerSessionId: "conv-123" });

    expect(result.ok).toBe(true);
    expect(result.envelope.session.provider_session_id).toBe("conv-123");
  });

  it("returns native_capture_unavailable for a mismatched/off-route/malformed native payload", async () => {
    const fetch = vi.fn(async (input) => {
      const url = new URL(String(input), "https://chatgpt.com");
      if (url.pathname === "/backend-api/conversation/conv-123") {
        // Mismatched conversation id in the payload body.
        return jsonResponse({ conversation_id: "other", mapping: {} });
      }
      return notFoundResponse();
    });
    const { sendRuntimeMessage } = installChatgpt({ url: "https://chatgpt.com/c/conv-123", fetch });

    const result = await sendRuntimeMessage({ type: "polylogue.capturePage", reason: "message_layer_save" });

    expect(result).toMatchObject({ ok: false, error: "native_capture_unavailable" });
    let repeated = result;
    for (let attempt = 0; attempt < 8; attempt++) repeated = await sendRuntimeMessage({ type: "polylogue.capturePage" });
    expect(repeated).toMatchObject({ ok: false, error: "native_capture_unavailable" });
    expect(repeated.native_attempts).toHaveLength(6);
    expect(repeated.native_attempts_dropped).toBeGreaterThan(0);
  });

  it("returns a typed bridge rate limit without fallback reads or asset downloads", async () => {
    const calls = [];
    const fetch = vi.fn(async (input) => {
      const url = new URL(String(input), "https://chatgpt.com");
      calls.push(url.pathname);
      if (url.pathname === "/backend-api/conversation/conv-429") {
        return new globalThis.Response(JSON.stringify({ detail: "rate limited" }), {
          status: 429,
          headers: { "content-type": "application/json", "Retry-After": "73" },
        });
      }
      return notFoundResponse();
    });
    const { dom, sendRuntimeMessage } = installChatgpt({ url: "https://chatgpt.com/c/conv-429", fetch });
    dom.window.dispatchEvent(new dom.window.MessageEvent("message", {
      source: dom.window,
      origin: dom.window.location.origin,
      data: {
        type: "polylogue.page.v2.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.chatgpt.nativeCapture",
        capture: {
          ok: true,
          status: 200,
          contentType: "application/json",
          url: "https://chatgpt.com/backend-api/conversation/conv-429",
          bodyRef: await dom.window.polylogueAssetStream.stageResponse(jsonResponse({
            conversation_id: "conv-429",
            current_node: "assistant",
            mapping: {
              assistant: {
                message: {
                  id: "assistant",
                  author: { role: "assistant" },
                  status: "finished_successfully",
                  content: { parts: ["[file](sandbox:/mnt/data/never-fetch.zip)"] },
                },
              },
            },
          }), "chatgpt", new globalThis.AbortController().signal),
        },
      },
    }));

    const result = await sendRuntimeMessage({
      type: "polylogue.capturePage",
      reason: "freshness_convergence",
      providerSessionId: "conv-429",
    });

    expect(result).toMatchObject({
      ok: false,
      error: "rate_limited",
      outcome: "rate_limited",
      retry_after_seconds: 73,
    });
    expect(calls.filter((path) => path === "/backend-api/conversation/conv-429")).toHaveLength(1);
    expect(calls).not.toContain("/backend-api/conversation/conv-429/interpreter/download");
  });

  it("keeps a rate limit typed when the provider omits Retry-After", async () => {
    const fetch = vi.fn(async (input) => {
      const url = new URL(String(input), "https://chatgpt.com");
      if (url.pathname === "/backend-api/conversation/conv-429-no-header") {
        return new globalThis.Response(JSON.stringify({ detail: "rate limited" }), {
          status: 429,
          headers: { "content-type": "application/json" },
        });
      }
      return notFoundResponse();
    });
    const { sendRuntimeMessage } = installChatgpt({
      url: "https://chatgpt.com/c/conv-429-no-header",
      fetch,
    });

    const result = await sendRuntimeMessage({
      type: "polylogue.capturePage",
      reason: "freshness_convergence",
      providerSessionId: "conv-429-no-header",
    });

    expect(result).toMatchObject({ ok: false, outcome: "rate_limited", retry_after_seconds: null });
    expect(fetch.mock.calls.filter(([input]) => String(input).includes("/backend-api/conversation/conv-429-no-header")))
      .toHaveLength(1);
  });

  it("records a manual rate limit before a second capture can fetch again", async () => {
    let cooldownUntil = 0;
    const runtimeMessages = [];
    const fetch = vi.fn(async (input) => {
      const url = new URL(String(input), "https://chatgpt.com");
      if (url.pathname === "/backend-api/conversation/conv-manual-429") {
        return new globalThis.Response(JSON.stringify({ detail: "rate limited" }), {
          status: 429,
          headers: { "content-type": "application/json", "Retry-After": "73" },
        });
      }
      return notFoundResponse();
    });
    const { sendRuntimeMessage } = installChatgpt({
      url: "https://chatgpt.com/c/conv-manual-429",
      fetch,
      runtimeMessage: async (message) => {
        runtimeMessages.push(message);
        if (message.type === "polylogue.providerThrottle") {
          return cooldownUntil > Date.now()
            ? { ok: false, outcome: "rate_limited", retry_after_seconds: Math.ceil((cooldownUntil - Date.now()) / 1000) }
            : { ok: true };
        }
        if (message.type === "polylogue.providerRateLimited") {
          const delay = message.retry_after_seconds !== undefined ? message.retry_after_seconds * 1000 : Number(message.retry_after) * 1000;
          cooldownUntil = Date.now() + delay;
          return { ok: true };
        }
        return undefined;
      },
    });

    const first = await sendRuntimeMessage({ type: "polylogue.capturePage", reason: "message_layer_save" });
    const second = await sendRuntimeMessage({ type: "polylogue.capturePage", reason: "message_layer_save" });

    expect(first).toMatchObject({ ok: false, outcome: "rate_limited", retry_after_seconds: 73 });
    expect(second).toMatchObject({ ok: false, outcome: "rate_limited" });
    expect(runtimeMessages).toContainEqual(expect.objectContaining({
      type: "polylogue.providerRateLimited",
      retry_after_seconds: 73,
    }));
    expect(fetch.mock.calls.filter(([input]) => String(input).includes("/backend-api/conversation/conv-manual-429")))
      .toHaveLength(1);
  });

  it("preserves a genuine forty-eight hour provider Retry-After", async () => {
    const runtimeMessages = [];
    const fetch = vi.fn(async (input) => {
      const url = new URL(String(input), "https://chatgpt.com");
      if (url.pathname === "/backend-api/conversation/conv-long-429") {
        return new globalThis.Response(JSON.stringify({ detail: "rate limited" }), {
          status: 429,
          headers: { "content-type": "application/json", "Retry-After": "172800" },
        });
      }
      return notFoundResponse();
    });
    const { sendRuntimeMessage } = installChatgpt({
      url: "https://chatgpt.com/c/conv-long-429",
      fetch,
      runtimeMessage: async (message) => {
        runtimeMessages.push(message);
        if (message.type === "polylogue.providerThrottle") return { ok: true };
        if (message.type === "polylogue.providerRateLimited") return { ok: true };
        return undefined;
      },
    });

    const providerCooldownSeconds = 48 * 60 * 60;
    const result = await sendRuntimeMessage({ type: "polylogue.capturePage", reason: "message_layer_save" });

    expect(result).toMatchObject({ ok: false, outcome: "rate_limited", retry_after_seconds: providerCooldownSeconds });
    expect(runtimeMessages).toContainEqual(expect.objectContaining({
      type: "polylogue.providerRateLimited",
      retry_after_seconds: providerCooldownSeconds,
    }));

  });

  it("fails closed when the shared throttle authority is unavailable", async () => {
    const fetch = vi.fn(async () => notFoundResponse());
    const { sendRuntimeMessage } = installChatgpt({
      url: "https://chatgpt.com/c/authority-unavailable",
      fetch,
      runtimeMessage: async (message) => {
        if (message.type === "polylogue.providerThrottle") throw new Error("runtime unavailable");
        return undefined;
      },
    });

    const result = await sendRuntimeMessage({ type: "polylogue.capturePage", reason: "message_layer_save" });

    expect(result).toMatchObject({ ok: false, error: "provider_throttle_authority_unavailable" });
    expect(fetch).not.toHaveBeenCalled();
  });

  it("fails closed when the shared throttle authority returns a negative response", async () => {
    const fetch = vi.fn(async () => notFoundResponse());
    const { sendRuntimeMessage } = installChatgpt({
      url: "https://chatgpt.com/c/authority-negative",
      fetch,
      runtimeMessage: async (message) => {
        if (message.type === "polylogue.providerThrottle") return { ok: false, error: "worker reloading" };
        return undefined;
      },
    });

    const result = await sendRuntimeMessage({ type: "polylogue.capturePage", reason: "message_layer_save" });

    expect(result).toMatchObject({ ok: false, error: "provider_throttle_authority_unavailable" });
    expect(fetch).not.toHaveBeenCalled();
  });
});

describe("ChatGPT canonical asset-plan consumption", () => {
  // Canonical pointer classification and sandbox punctuation/deduplication:
  // test_parsers_chatgpt.py audio/pointer and sandbox-link controls. These
  // tests exercise the content bridge consuming that parser's descriptors.
  it("acquires a canonical audio descriptor through its exact provider file ID", async () => {
    const fetch = vi.fn(async (input) => {
      const url = new URL(String(input), "https://chatgpt.com");
      if (url.pathname === "/api/auth/session") return jsonResponse({ accessToken: "synthetic-asset-token" });
      if (url.pathname === "/backend-api/files/file-TRANSCRIBE8/download") {
        return jsonResponse({ download_url: "https://chatgpt.com/transcription-bytes", file_name: "transcription.wav" });
      }
      if (url.pathname === "/transcription-bytes") {
        return new globalThis.Response(new Uint8Array([1, 2, 3]), { headers: { "content-type": "audio/wav" } });
      }
      return notFoundResponse();
    });
    const { dom, sendRuntimeMessage } = installChatgpt({ fetch });
    const descriptor = { provider_attachment_id: "file-service://file-TRANSCRIBE8", attachment_kind: "audio",
      name: "transcription.wav", mime_type: "audio/wav", provider_meta: { provider_file_id: "file-TRANSCRIBE8" } };
    const result = await sendRuntimeMessage({ type: "polylogue.acquireRecordAssets", provider: "chatgpt",
      nativeId: "conversation-1", capture_ref: "original-raw", recordKey: "original-node", attachmentOrdinal: 4, attachments: [descriptor] });
    expect(result.ok).toBe(true);
    expect(result.acquisition.outcome).toMatchObject({ attempted: 1, acquired: 1, failed: [] });
    const acquired = result.acquisition.attachments[0];
    expect(acquired).toMatchObject({ provider_attachment_id: descriptor.provider_attachment_id, attachment_kind: "audio" });
    expect((await dom.__captureRuntime.staging.file(acquired.staged_asset.id)).size).toBe(3);
    expect(fetch.mock.calls.some(([url]) => String(url).includes("/backend-api/files/file-TRANSCRIBE8/download"))).toBe(true);
  });

  it("keeps duplicate provider IDs in distinct canonical occurrence acquisitions", async () => {
    const { dom, sendRuntimeMessage } = installChatgpt({ fetch: async (input) => {
      const url = new URL(String(input), "https://chatgpt.com");
      if (url.pathname === "/api/auth/session") return jsonResponse({ accessToken: "synthetic-asset-token" });
      if (url.pathname === "/backend-api/files/file-SHARED/download") return jsonResponse({ download_url: "https://chatgpt.com/shared-bytes" });
      if (url.pathname === "/shared-bytes") return new globalThis.Response("shared evidence", { headers: { "content-type": "text/plain" } });
      return notFoundResponse();
    } });
    const descriptor = { provider_attachment_id: "file-service://file-SHARED", name: "shared.txt",
      provider_meta: { provider_file_id: "file-SHARED" } };
    const acquired = [];
    for (const ordinal of [0, 1]) {
      const result = await sendRuntimeMessage({ type: "polylogue.acquireRecordAssets", provider: "chatgpt", nativeId: "conversation-1",
        capture_ref: "original-raw", recordKey: "same-raw-node", attachmentOrdinal: ordinal, attachments: [descriptor] });
      expect(result.ok).toBe(true);
      acquired.push(result.acquisition.attachments[0]);
    }
    expect(acquired[0].staged_asset.id).not.toBe(acquired[1].staged_asset.id);
    for (const asset of acquired) expect(await (await dom.__captureRuntime.staging.file(asset.staged_asset.id)).text()).toBe("shared evidence");
  });
});

describe("ChatGPT continuous generation observation", () => {
  it("samples generation controls while transcript mutations never reach the trailing debounce", async () => {
    vi.useFakeTimers();
    const hints = [];
    const { dom } = installChatgpt({ runtimeMessage: async message => {
      if (message.type === "polylogue.captureFreshnessHint") hints.push(message);
      return { ok: true };
    } });
    let now = 100000;
    dom.window.Date.now = () => now;
    const article = dom.window.document.createElement("article");
    article.setAttribute("data-turn", "assistant"); article.setAttribute("data-turn-id", "turn-1");
    const stop = dom.window.document.createElement("button"); stop.setAttribute("data-testid", "stop-button");
    const text = dom.window.document.createElement("span"); article.append(stop, text);
    dom.window.document.body.append(article);
    try {
      for (let tick = 0; tick < 50; tick++) {
        text.textContent = `Neutral streamed fragment ${tick}`;
        await Promise.resolve();
        now += 100;
        await vi.advanceTimersByTimeAsync(100);
      }
      expect(hints.flatMap(hint => hint.generation_observations || []).map(observation => observation.state)).toContain("started");
      stop.remove(); await Promise.resolve();
      await vi.advanceTimersByTimeAsync(1000);
      expect(hints.flatMap(hint => hint.generation_observations || []).map(observation => observation.state)).toContain("completed");
      expect(hints.every(hint => hint.provider_session_id === "conversation-1")).toBe(true);
      const observations = () => hints.flatMap(hint => hint.generation_observations || []);
      const beforeHide = observations().map(observation => observation.observation_id);
      dom.window.dispatchEvent(new dom.window.Event("pagehide"));
      article.append(stop); text.textContent = "Mutation while page observation is suspended";
      await Promise.resolve(); await vi.advanceTimersByTimeAsync(1000);
      // A previously observed prose hint may still settle; no new generation
      // observation may be produced while this page owner is suspended.
      expect(observations().map(observation => observation.observation_id)).toEqual(beforeHide);
      dom.window.dispatchEvent(new dom.window.Event("pageshow"));
      await vi.advanceTimersByTimeAsync(1000);
      const started = observations().filter(observation => observation.state === "started");
      expect(started).toHaveLength(2);
      expect(started[1].observation_id).not.toBe(started[0].observation_id);
      expect(started[1].wall_elapsed_ms).toBe(0);
    } finally {
      dom.window.dispatchEvent(new dom.window.Event("pagehide"));
      dom.window.close(); vi.useRealTimers();
    }
  });
});

describe("chatgpt.js capture-owned native reads do not re-enter the freshness queue", () => {
  // polylogue-6nzro: this is the actual feedback-loop cut. The bridge tags its
  // own conversation read `source: "polylogue_native_fetch"` before
  // re-broadcasting it, and the content script refuses to schedule a freshness
  // hint for a read polylogue itself caused. Without this pair every capture
  // observes itself and schedules the next one, which is the 13-15s recapture
  // storm the bead reports.
  //
  // Anti-vacuity: delete the `&& data.capture.source !== "polylogue_native_fetch"`
  // clause in src/content/chatgpt.js, or the `source:` tag in
  // src/content/chatgpt_bridge.js, and the first test below goes red.
  async function nativeCaptureEvent(dom, { source = null } = {}) {
    const capture = {
      ok: true,
      url: "https://chatgpt.com/backend-api/conversation/conversation-1",
      bodyRef: await dom.window.polylogueAssetStream.stageResponse(
        jsonResponse({ conversation_id: "conversation-1", mapping: {}, update_time: 1_700_000_000 }), "chatgpt", new globalThis.AbortController().signal,
      ),
    };
    if (source) capture.source = source;
    dom.window.postMessage({ type: "polylogue.page.v2.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.chatgpt.nativeCapture", capture });
  }

  function collectHints() {
    const hints = [];
    const runtimeMessage = async (message) => {
      if (message.type !== "polylogue.captureFreshnessHint") return undefined;
      hints.push(message);
      return { ok: true };
    };
    return { hints, runtimeMessage };
  }

  // The content script debounces a freshness hint for 750ms before sending it.
  const afterHintDebounce = () => new Promise((resolve) => globalThis.setTimeout(resolve, 1200));

  it("schedules no freshness hint for a capture-initiated bridge read", async () => {
    const { hints, runtimeMessage } = collectHints();
    const { dom } = installChatgpt({ runtimeMessage });

    await nativeCaptureEvent(dom, { source: "polylogue_native_fetch" });
    await afterHintDebounce();

    expect(hints).toEqual([]);
  });

  it("still schedules a freshness hint for an organic provider read", async () => {
    const { hints, runtimeMessage } = collectHints();
    const { dom } = installChatgpt({ runtimeMessage });

    await nativeCaptureEvent(dom);
    await afterHintDebounce();

    expect(hints).toMatchObject([
      { reason: "provider_native_observed", provider: "chatgpt", provider_session_id: "conversation-1" },
    ]);
  });
});

describe("chatgpt.js capture-initiated native fetch does not observe itself", () => {
  // polylogue-6nzro: the end-to-end form of the loop cut. A real capture drives
  // the MAIN-world bridge's fetchConversation, which calls
  // remember({ ...capture, source: "polylogue_native_fetch" }); remember() then
  // re-broadcasts that capture as a nativeCapture message, straight back into
  // the content script's own listener. If either half of the pair is missing,
  // the capture observes itself and queues a provider_native_observed hint --
  // the next capture then does the same, which is the recapture storm.
  //
  // Anti-vacuity: this test goes red under EITHER mutation -- dropping the
  // `source:` tag in src/content/chatgpt_bridge.js, or dropping the
  // `data.capture.source !== "polylogue_native_fetch"` clause in
  // src/content/chatgpt.js.
  it("emits no freshness hint for the conversation it just captured", async () => {
    const hints = [];
    const fetch = vi.fn(async (input) => {
      const url = new URL(String(input), "https://chatgpt.com");
      if (url.pathname === "/backend-api/conversation/conv-loop") {
        return jsonResponse({
          conversation_id: "conv-loop",
          update_time: 1_700_000_000,
          mapping: {
            node: {
              id: "node",
              parent: null,
              message: { id: "m", author: { role: "user" }, content: { parts: ["hello"] } },
            },
          },
        });
      }
      return notFoundResponse();
    });
    const { sendRuntimeMessage } = installChatgpt({
      url: "https://chatgpt.com/c/conv-loop",
      fetch,
      runtimeMessage: async (message) => {
        if (message.type !== "polylogue.captureFreshnessHint") return undefined;
        hints.push(message);
        return { ok: true };
      },
    });

    const result = await sendRuntimeMessage({
      type: "polylogue.capturePage",
      reason: "completion_monitor",
      providerSessionId: "conv-loop",
    });
    expect(result).toMatchObject({ ok: true });

    // Past the content script's 750ms freshness-hint debounce.
    await new Promise((resolve) => globalThis.setTimeout(resolve, 1200));

    expect(hints).toEqual([]);
  });
});


it.each(["success", "failure", "cancel"])("retains exact native progress without settling the response early: %s", async (outcome) => {
  let entered = false;
  let finish;
  let requestId;
  const fetch = vi.fn((input, options = {}) => {
    const url = new URL(String(input));
    if (url.pathname === "/api/auth/session") return new Promise((resolve, reject) => {
      entered = true;
      finish = () => outcome === "failure" ? reject(new Error("synthetic auth fault")) : resolve(notFoundResponse());
      options.signal.addEventListener("abort", () => reject(new globalThis.DOMException("capture_cancelled", "AbortError")), { once: true });
    });
    if (outcome === "failure") return Promise.reject(new Error("private provider fault"));
    return Promise.resolve(jsonResponse({ id: "conversation-1", mapping: {} }));
  });
  const { dom, sendRuntimeMessage } = installChatgpt({ fetch });
  const progressFrames = [];
  dom.window.addEventListener("message", event => {
    if (event.data?.type === "polylogue.page.v2.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.chatgpt.nativeFetchRequest") requestId = event.data.requestId;
    if (event.data?.progress) progressFrames.push(event.data);
  });
  let settled = false;
  const capture = sendRuntimeMessage({ type: "polylogue.capturePage", deferReceiver: true }).then(result => { settled = true; return result; });
  await vi.waitFor(() => expect(entered).toBe(true));
  expect(settled).toBe(false);
  expect(progressFrames.map(frame => frame.progress)).toEqual([
    { stage: "staging", state: "BEGIN" }, { stage: "staging", state: "END" }, { stage: "provider_auth", state: "BEGIN" },
  ]);
  for (const frame of [
    { type: "polylogue.page.v2.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.chatgpt.nativeFetchResponse", requestId: "unrelated", progress: { stage: "body", state: "END" } },
    { type: "polylogue.page.v2.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.chatgpt.nativeFetchResponse", requestId, progress: { stage: "https://private.invalid/token", state: "BEGIN" } },
    { type: "polylogue.page.v2.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.chatgpt.nativeFetchResponse", requestId, progress: { stage: "body", state: "END", token: "private-token" } },
    { type: "polylogue.page.v2.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.chatgpt.nativeFetchResponse", requestId, progress: { stage: "body", state: "END" }, capture: { ok: true } },
  ]) dom.window.postMessage(frame);
  await new Promise(resolve => dom.window.setTimeout(resolve, 0));
  expect(settled).toBe(false);
  if (outcome === "cancel") await sendRuntimeMessage({ type: "polylogue.cancelCapture" });
  else finish();
  const result = await capture;
  expect(result.native_progress.length).toBeLessThanOrEqual(6);
  expect(JSON.stringify(result.native_progress)).not.toContain("private");
  expect(result.native_progress).not.toContainEqual({ stage: "body", state: "END", token: "private-token" });
  if (outcome === "cancel") {
    expect(result).toMatchObject({ ok: false, outcome: "cancelled" });
    expect(result.native_progress.at(-1)).toEqual({ stage: "provider_auth", state: "BEGIN" });
  } else if (outcome === "failure") {
    expect(result).toMatchObject({ ok: false, error: "native_capture_unavailable" });
    expect(result.native_progress.at(-1)).toEqual({ stage: "header", state: "END" });
  } else {
    expect(result.ok).toBe(true);
    expect(result.native_progress.at(-1)).toEqual({ stage: "canonical", state: "END" });
    // The provider's existing auth fallback and original terminal capture remain unchanged.
    expect(result.envelope.session.provider_session_id).toBe("conversation-1");
  }
  const terminalCount = progressFrames.length;
  await new Promise(resolve => dom.window.setTimeout(resolve, 0));
  expect(progressFrames.length).toBe(terminalCount);
});


it("retains a pending native-header marker until original cancellation drains", async () => {
  let finishHeader;
  let headerEntered = false;
  const { sendRuntimeMessage } = installChatgpt({
    fetch: async input => new URL(String(input)).pathname === "/api/auth/session"
      ? notFoundResponse() : jsonResponse({ id: "conversation-1", mapping: {} }),
    runtimeMessage: message => message.type === "polylogue.nativeCaptureHeader" ? new Promise(resolve => {
      headerEntered = true; finishHeader = resolve;
    }) : undefined,
  });
  const capture = sendRuntimeMessage({ type: "polylogue.capturePage" });
  await vi.waitFor(() => expect(headerEntered).toBe(true));
  let cancelled = false;
  const cancellation = sendRuntimeMessage({ type: "polylogue.cancelCapture" }).then(result => { cancelled = true; return result; });
  await Promise.resolve();
  expect(cancelled).toBe(false);
  finishHeader();
  await cancellation;
  const result = await capture;
  expect(result).toMatchObject({ ok: false, outcome: "cancelled" });
  expect(result.native_progress.at(-1)).toEqual({ stage: "header", state: "BEGIN" });
});


it.each(["success", "cancel"])("marks the actual suspended native body before settlement: %s", async (outcome) => {
  let bodyController;
  let bodyCancelled = false;
  const frames = [];
  const { dom, sendRuntimeMessage } = installChatgpt({ fetch: async input => {
    if (new URL(String(input)).pathname === "/api/auth/session") return notFoundResponse();
    return new globalThis.Response(new globalThis.ReadableStream({
      start(controller) { bodyController = controller; },
      cancel() { bodyCancelled = true; },
    }), { headers: { "content-type": "application/json" } });
  } });
  dom.window.addEventListener("message", event => {
    if (event.data?.progress) frames.push(event.data.progress);
  });
  const capture = sendRuntimeMessage({ type: "polylogue.capturePage", deferReceiver: true });
  await vi.waitFor(() => expect(frames.at(-1)).toEqual({ stage: "body", state: "BEGIN" }));
  expect(bodyController).toBeDefined();
  if (outcome === "cancel") await sendRuntimeMessage({ type: "polylogue.cancelCapture" });
  else {
    bodyController.enqueue(new TextEncoder().encode(JSON.stringify({ id: "conversation-1", mapping: {} })));
    bodyController.close();
  }
  const result = await capture;
  if (outcome === "cancel") {
    expect(result).toMatchObject({ ok: false, outcome: "cancelled" });
    expect(result.native_progress.at(-1)).toEqual({ stage: "body", state: "BEGIN" });
    expect(bodyCancelled).toBe(true);
    expect(frames).not.toContainEqual({ stage: "body", state: "END" });
  } else {
    expect(result.ok).toBe(true);
    expect(frames).toContainEqual({ stage: "body", state: "END" });
    expect(result.native_progress.at(-1)).toEqual({ stage: "canonical", state: "END" });
  }
});

it.each([
  ["throttle", "polylogue.providerThrottle"],
  ["restore", "polylogue.restoreNativeCapture"],
  ["canonical", "polylogue.normalizeNativeCapture"],
])("freezes the exact original %s await on cancellation", async (stage, messageType) => {
  let release;
  const { sendRuntimeMessage } = installChatgpt({
    fetch: async input => new URL(String(input)).pathname === "/api/auth/session"
      ? notFoundResponse() : jsonResponse({ id: "conversation-1", mapping: {} }),
    runtimeMessage: message => message.type === messageType ? new Promise(resolve => { release = resolve; }) : undefined,
  });
  const capture = sendRuntimeMessage({ type: "polylogue.capturePage", deferReceiver: true });
  await vi.waitFor(() => expect(release).toBeDefined());
  const cancellation = sendRuntimeMessage({ type: "polylogue.cancelCapture" });
  release();
  await cancellation;
  const result = await capture;
  expect(result).toMatchObject({ ok: false, outcome: "cancelled" });
  expect(result.native_progress.at(-1)).toEqual({ stage, state: "BEGIN" });
});


it.each(["success", "failure", "cancel", "ambiguous", "observer_failure"])("filters original private preparation progress and physically removes observer: %s", async outcome => {
  let release; let rawRef; let nativeRequestId;
  const {sendRuntimeMessage, storageListeners, emitStorage} = installChatgpt({
    storageFailure: outcome === "observer_failure",
    fetch: async input => new URL(String(input)).pathname === "/api/auth/session" ? notFoundResponse() : jsonResponse({id:"conversation-1",mapping:{}}),
    runtimeMessage: message => message.type === "polylogue.normalizeNativeCapture" ? new Promise((resolve,reject)=>{rawRef=message.raw_ref;nativeRequestId=message.native_request_id;release=()=>outcome==="failure"?reject(new Error("synthetic preparation refusal")):resolve();}) : undefined,
  });
  const capture=sendRuntimeMessage({type:"polylogue.capturePage",deferReceiver:true});
  await vi.waitFor(()=>expect(release).toBeDefined());
  expect(storageListeners.size).toBe(outcome === "observer_failure" ? 0 : 2);
  expect(nativeRequestId).toMatch(/^polylogue-native-fetch-\d+-[a-z0-9]+$/);
  const marker=(phase,state,extra={})=>({at:new Date().toISOString(),stage:"native_preparation_progress",phase,state,acquisition_ref:rawRef.id,native_request_id:nativeRequestId,...extra});
  const historical=marker("normalize_admission","BEGIN");
  emitStorage({polylogueDebugLog:{oldValue:[historical],newValue:[historical]}});
  emitStorage({polylogueDebugLog:{oldValue:[],newValue:[marker("native_prepare","BEGIN")]}}); // No fresh admission.
  emitStorage({polylogueDebugLog:{oldValue:[],newValue:[marker("normalize_admission","BEGIN",{acquisition_ref:"other"})]}});
  emitStorage({polylogueDebugLog:{oldValue:[],newValue:[marker("normalize_admission","BEGIN",{native_request_id:"polylogue-native-fetch-1-foreign"})]}});
  emitStorage({polylogueDebugLog:{oldValue:[],newValue:[marker("normalize_admission","BEGIN",{native_request_id:undefined})]}});
  emitStorage({polylogueDebugLog:{oldValue:[],newValue:[marker("normalize_admission","BEGIN",{native_request_id:"staging-uuid"})]}});
  emitStorage({polylogueDebugLog:{oldValue:[],newValue:[marker("normalize_admission","BEGIN",{started_at:Date.now()+1000})]}});
  const admission=marker("normalize_admission","BEGIN");
  // Use an actually fresh row, not retained matching fields.
  admission.at="2099-01-01T00:00:01.000Z";
  emitStorage({polylogueDebugLog:{oldValue:[],newValue:[admission]}});
  const preparation=marker("native_prepare","BEGIN");
  emitStorage({polylogueDebugLog:{oldValue:[admission],newValue:[preparation,admission]}});
  emitStorage({polylogueDebugLog:{oldValue:[],newValue:[marker("native_prepare","END",{token:"private"})]}});
  if(outcome==="ambiguous") emitStorage({polylogueDebugLog:{oldValue:[],newValue:[marker("normalize_admission","BEGIN",{at:"2099-01-01T00:00:02.000Z"})]}});
  let cancellation;
  if(outcome==="cancel") cancellation=sendRuntimeMessage({type:"polylogue.cancelCapture"});
  release();
  const result=await capture;
  if(cancellation) await cancellation;
  expect(storageListeners.size).toBe(outcome === "observer_failure" ? 0 : 1);
  const rows=result.native_progress.filter(row=>row.source==="background_debug_log");
  expect(rows).toEqual(["ambiguous","observer_failure"].includes(outcome)?[]:[{stage:"normalize_admission",state:"BEGIN",source:"background_debug_log"},{stage:"native_prepare",state:"BEGIN",source:"background_debug_log"}]);
  emitStorage({polylogueDebugLog:{oldValue:[],newValue:[marker("native_prepare","END")]}});
  expect(result.native_progress.filter(row=>row.source==="background_debug_log")).toEqual(rows);
  expect(JSON.stringify(result.native_progress)).not.toContain(rawRef.id);
  expect(JSON.stringify(result.native_progress)).not.toContain(nativeRequestId);
});
