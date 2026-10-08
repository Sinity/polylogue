import { stagingRuntime, attachmentBytes } from "../infra/capture-staging.js";
import { Buffer } from "node:buffer";
import { createHash, webcrypto } from "node:crypto";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { TextEncoder } from "node:util";
import { fileURLToPath } from "node:url";
import { Script } from "node:vm";

import { JSDOM } from "jsdom";
import { afterEach, describe, expect, it, vi } from "vitest";

const testDirectory = dirname(fileURLToPath(import.meta.url));
const assetStreamSource = readFileSync(resolve(testDirectory, "../../src/content/asset_stream.js"), "utf8");
const bridgeSource = readFileSync(resolve(testDirectory, "../../src/content/grok_bridge.js"), "utf8");
const conversationId = "1f9de430-6505-4d43-935b-ec0dd1c13222";
const assetBytes = new TextEncoder().encode("polylogue grok asset fixture\n");
const expectedSha256 = createHash("sha256").update(assetBytes).digest("hex");
const openDoms = [];

function jsonResponse(body, status = 200) {
  return new globalThis.Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
}

function byteResponse(bytes, status = 200) {
  return new globalThis.Response(bytes, { status, headers: { "content-type": "text/plain" } });
}

function conversationMetadata(overrides = {}) {
  return { conversationId, title: "Fixture conversation", temporary: false, createTime: "2026-06-27T12:39:08Z", modifyTime: "2026-06-27T13:46:12Z", ...overrides };
}

function responsesPayload() {
  return {
    responses: [
      { responseId: "r-1", sender: "human", message: "hello", createTime: "2026-06-27T12:39:09Z" },
      { responseId: "r-2", sender: "ASSISTANT", parentResponseId: "r-1", message: "hi there", createTime: "2026-06-27T12:39:15Z" },
    ],
  };
}

function makeDom(fetchImpl, url = `https://grok.com/c/${conversationId}`) {
  const dom = new JSDOM("<!doctype html><title>Grok fixture</title>", { url, runScripts: "outside-only" });
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
  Object.defineProperty(dom.window, "fetch", { configurable: true, value: fetchImpl });
  return dom;
}

function installBridge(fetchImpl) {
  const dom = makeDom(fetchImpl);
  const pending = new Map();
  const posted = [];
  dom.__captureRuntime = stagingRuntime(undefined, { tab_id: 42, document_id: "synthetic-document", provider: "grok" });
  Object.defineProperty(dom.window, "chrome", { configurable: true, value: { storage: { local: { get: async () => ({ polylogueAmbientSettings: {}, polylogueReceiverPairing: { receiver_id: "neutral", state: "online" } }) } }, runtime: { id: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", sendMessage: dom.__captureRuntime.sendMessage } } });
  Object.defineProperty(dom.window, "postMessage", {
    configurable: true,
    value(data) {
      posted.push(data);
      dom.window.queueMicrotask(() => dom.window.dispatchEvent(new dom.window.MessageEvent("message", { source: dom.window, origin: dom.window.location.origin, data })));
      const resolve = pending.get(data?.requestId);
      if (resolve && (data?.type === "polylogue.page.v2.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.grok.nativeFetchResponse" || data?.type === "polylogue.page.v2.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.grok.assetFetchResponse")) {
        pending.delete(data.requestId);
        resolve(data);
      }
    },
  });
  new Script(assetStreamSource).runInContext(dom.getInternalVMContext());
  new Script(bridgeSource).runInContext(dom.getInternalVMContext());

  function requestNative(overrides = {}) {
    const requestId = overrides.requestId || `native-request-${pending.size + 1}-${posted.length}`;
    const response = new Promise((resolve) => pending.set(requestId, resolve));
    dom.window.dispatchEvent(new dom.window.MessageEvent("message", {
      source: dom.window,
      origin: dom.window.location.origin,
      data: { type: "polylogue.page.v2.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.grok.nativeFetchRequest", requestId, conversationId, ...overrides },
    }));
    return response;
  }

  function requestAsset(overrides = {}) {
    const requestId = `asset-request-${pending.size + 1}-${posted.length}`;
    const response = new Promise((resolve) => pending.set(requestId, resolve));
    return dom.window.polylogueAssetStream.request({ provider: "grok", requestId, signal: new dom.window.AbortController().signal, start: () => {
      dom.window.dispatchEvent(new dom.window.MessageEvent("message", {
      source: dom.window,
      origin: dom.window.location.origin,
      data: { type: "polylogue.page.v2.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.grok.assetFetchRequest", requestId, request: { key: "users/u1/asset-1/content", ...overrides } },
    }));
      return response.then((message) => message.outcome);
    } }).then((outcome) => ({ outcome }));
  }

  return { dom, posted, requestNative, requestAsset };
}

afterEach(() => {
  for (const dom of openDoms.splice(0)) dom.window.close();
});

describe("Grok bridge conversation fetch contract", () => {
  it("posts a native failure reply after an actual provider fetch rejection", async () => {
    const fetch = vi.fn(async () => { throw new TypeError("synthetic_native_fetch_failed"); });
    const { requestNative, posted } = installBridge(fetch);
    expect(await requestNative({ requestId: "native-failed" })).toMatchObject({
      type: "polylogue.page.v2.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.grok.nativeFetchResponse", requestId: "native-failed", error: "synthetic_native_fetch_failed",
    });
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(posted.filter((message) => message.type === "polylogue.page.v2.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.grok.nativeFetchResponse")).toHaveLength(1);
  });

  it("drains a cancelled native fetch and posts its exact request failure reply", async () => {
    let started; let drained = false;
    const began = new Promise((resolve) => { started = resolve; });
    const fetch = vi.fn(async (_input, { signal }) => {
      started();
      try { await new Promise((resolve, reject) => {
        signal.addEventListener("abort", () => reject(signal.reason), { once: true });
        if (signal.aborted) reject(signal.reason);
      }); } finally { drained = true; }
    });
    const { dom, requestNative, posted } = installBridge(fetch);
    const request = requestNative({ requestId: "native-cancelled" });
    await began;
    dom.window.postMessage({ type: "polylogue.page.v2.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.grok.cancelRequest", requestId: "native-cancelled" }, dom.window.location.origin);
    expect(await request).toMatchObject({ type: "polylogue.page.v2.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.grok.nativeFetchResponse", requestId: "native-cancelled", error: "capture_cancelled" });
    expect(drained).toBe(true);
    expect(posted.filter((message) => message.type === "polylogue.page.v2.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.grok.nativeFetchResponse")).toHaveLength(1);
  });

  it("refuses staging admission before any Grok provider request when physical storage is unavailable", async () => {
    const fetch = vi.fn(async () => jsonResponse(conversationMetadata()));
    const { dom, requestNative } = installBridge(fetch);
    const begin = dom.__captureRuntime.staging.begin.bind(dom.__captureRuntime.staging);
    dom.__captureRuntime.staging.begin = async (owner, purpose, producerId) => {
      if (purpose.capture_bundle) throw new dom.window.DOMException("storage full", "QuotaExceededError");
      return begin(owner, purpose, producerId);
    };
    const result = await requestNative();
    expect(result).toHaveProperty("error");
    expect(fetch).not.toHaveBeenCalled();
    const bundles = [];
    for await (const row of dom.__captureRuntime.store.captures()) if (row.kind === "native-bundle") bundles.push(row);
    expect(bundles).toHaveLength(1);
    expect(bundles[0].native_id).toBe(conversationId);
    expect(bundles[0].replies).toEqual({});
  });

  it("retains named original conversation, response and inflight replies in one capture bundle", async () => {
    const fetch = vi.fn(async (input) => {
      const url = new URL(String(input), "https://grok.com");
      if (url.pathname === `/rest/app-chat/conversations/${conversationId}`) return jsonResponse(conversationMetadata());
      if (url.pathname === `/rest/app-chat/conversations/${conversationId}/responses`) return jsonResponse(responsesPayload());
      if (url.pathname === `/rest/app-chat/conversations/${conversationId}/response-node`) {
        return jsonResponse({ responseNodes: [], inflightResponses: [{ responseId: "r-3", sender: "ASSISTANT" }] });
      }
      throw new Error(`unexpected request: ${url.pathname}`);
    });
    const { requestNative, dom } = installBridge(fetch);

    const result = await requestNative();
    expect(result.capture.ok).toBe(true);
    const stage = dom.__captureRuntime.staging;
    const body = JSON.parse(await (await stage.file(result.capture.bodyRef.id)).text());
    const conversation = JSON.parse(await (await stage.file(result.capture.relatedRefs.conversation.id)).text());
    const nodes = JSON.parse(await (await stage.file(result.capture.relatedRefs.response_nodes.id)).text());
    expect(conversation.conversationId).toBe(conversationId);
    expect(body.responses).toHaveLength(2); expect(body.responses[1].message).toBe("hi there");
    expect(nodes.inflightResponses).toEqual([{ responseId: "r-3", sender: "ASSISTANT" }]);
    // Every request must carry the page's own session cookies.
    for (const call of fetch.mock.calls) {
      expect(call[1].credentials).toBe("include");
    }
  });

  it("fails the whole capture when /responses cannot be fetched, but not when /response-node fails", async () => {
    const fetch = vi.fn(async (input) => {
      const url = new URL(String(input), "https://grok.com");
      if (url.pathname === `/rest/app-chat/conversations/${conversationId}`) return jsonResponse(conversationMetadata());
      if (url.pathname === `/rest/app-chat/conversations/${conversationId}/responses`) return jsonResponse({ code: 5, message: "Not Found" }, 404);
      if (url.pathname === `/rest/app-chat/conversations/${conversationId}/response-node`) throw new Error("network down");
      throw new Error(`unexpected request: ${url.pathname}`);
    });
    const { requestNative } = installBridge(fetch);

    const result = await requestNative();
    expect(result.capture.ok).toBe(false);
    expect(result.capture.error).toBe("conversation_responses_fetch_failed");
  });

  it("does not fail the capture when response-node (inflight skeleton) is unavailable", async () => {
    const fetch = vi.fn(async (input) => {
      const url = new URL(String(input), "https://grok.com");
      if (url.pathname === `/rest/app-chat/conversations/${conversationId}`) return jsonResponse(conversationMetadata());
      if (url.pathname === `/rest/app-chat/conversations/${conversationId}/responses`) return jsonResponse(responsesPayload());
      if (url.pathname === `/rest/app-chat/conversations/${conversationId}/response-node`) return jsonResponse({ code: 5 }, 500);
      throw new Error(`unexpected request: ${url.pathname}`);
    });
    const { requestNative } = installBridge(fetch);

    const result = await requestNative();
    expect(result.capture.ok).toBe(true);
    expect(result.capture.relatedRefs).not.toHaveProperty("response_nodes");
    expect(result.capture.acquisition.response_node_status).toBe(500);
  });
});

describe("Grok bridge asset acquisition (assets.grok.com requires the grok.com session)", () => {
  it("acquires bytes with credentials:include and verifies sha256", async () => {
    const fetch = vi.fn(async (input, options = {}) => {
      const url = new URL(String(input));
      expect(url.hostname).toBe("assets.grok.com");
      expect(options.credentials).toBe("include");
      return byteResponse(assetBytes);
    });
    const { requestAsset, dom } = installBridge(fetch);

    const result = await requestAsset();
    expect(result.outcome.status).toBe("acquired");
    expect(result.outcome.asset.sha256).toBe(expectedSha256);
    expect(Buffer.from(await attachmentBytes(dom.__captureRuntime.staging, result.outcome.asset)).toString("utf8")).toBe("polylogue grok asset fixture\n");
  });

  it("preserves all acquired asset bytes without a byte refusal", async () => {
    const fetch = vi.fn(async () => byteResponse(assetBytes));
    const { requestAsset, dom } = installBridge(fetch);

    const result = await requestAsset();
    expect(result.outcome.status).toBe("acquired");
    expect(Buffer.from(await attachmentBytes(dom.__captureRuntime.staging, result.outcome.asset))).toEqual(Buffer.from(assetBytes));
  });

  it("classifies a 403 (verified live: assets.grok.com without credentials) as signed_url_expired", async () => {
    const fetch = vi.fn(async () => new globalThis.Response("", { status: 403 }));
    const { requestAsset } = installBridge(fetch);

    const result = await requestAsset();
    expect(result.outcome.status).toBe("signed_url_expired");
    expect(result.outcome.http_status).toBe(403);
  });
});

describe("Grok bridge asset host pinning", () => {
  // The asset key comes from the provider response and is therefore
  // attacker-influenced. The URL parser resolves a backslash in a
  // special-scheme path as a slash, so `\host/x` parses as an https URL whose
  // host is `host`. Anti-vacuity: restore the `assetUrl.protocol !== "https:"`
  // check and this goes green on the request actually being issued, with the
  // credentialed fetch reaching the foreign host below.
  it("refuses a backslash key that would resolve off assets.grok.com", async () => {
    const fetch = vi.fn(async () => byteResponse(assetBytes));
    const { requestAsset } = installBridge(fetch);

    const result = await requestAsset({ key: "\\attacker.example/steal" });
    expect(result.outcome.status).toBe("invalid_request");
    expect(result.outcome.detail).toBe("asset_key_invalid");
    expect(fetch).not.toHaveBeenCalled();
  });

  it("still accepts an ordinary provider asset key", async () => {
    const fetch = vi.fn(async (input) => {
      expect(new URL(String(input)).origin).toBe("https://assets.grok.com");
      return byteResponse(assetBytes);
    });
    const { requestAsset } = installBridge(fetch);

    const result = await requestAsset({ key: "/users/u1/asset-1/content" });
    expect(result.outcome.status).toBe("acquired");
  });
});
