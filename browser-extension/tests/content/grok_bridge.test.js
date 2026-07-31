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
  Object.defineProperty(dom.window, "postMessage", {
    configurable: true,
    value(data) {
      posted.push(data);
      const resolve = pending.get(data?.requestId);
      if (resolve && (data?.type === "polylogue.grok.nativeFetchResponse" || data?.type === "polylogue.grok.assetFetchResponse")) {
        pending.delete(data.requestId);
        resolve(data);
      }
    },
  });
  new Script(bridgeSource).runInContext(dom.getInternalVMContext());

  function requestNative(overrides = {}) {
    const requestId = `native-request-${pending.size + 1}-${posted.length}`;
    const response = new Promise((resolve) => pending.set(requestId, resolve));
    dom.window.dispatchEvent(new dom.window.MessageEvent("message", {
      source: dom.window,
      origin: dom.window.location.origin,
      data: { type: "polylogue.grok.nativeFetchRequest", requestId, conversationId, ...overrides },
    }));
    return response;
  }

  function requestAsset(overrides = {}) {
    const requestId = `asset-request-${pending.size + 1}-${posted.length}`;
    const response = new Promise((resolve) => pending.set(requestId, resolve));
    dom.window.dispatchEvent(new dom.window.MessageEvent("message", {
      source: dom.window,
      origin: dom.window.location.origin,
      data: { type: "polylogue.grok.assetFetchRequest", requestId, request: { key: "users/u1/asset-1/content", maxBytes: 1024, ...overrides } },
    }));
    return response;
  }

  return { dom, posted, requestNative, requestAsset };
}

afterEach(() => {
  for (const dom of openDoms.splice(0)) dom.window.close();
});

describe("Grok bridge conversation fetch contract", () => {
  it("combines conversation metadata, responses, and inflight skeleton into one capture", async () => {
    const fetch = vi.fn(async (input) => {
      const url = new URL(String(input), "https://grok.com");
      if (url.pathname === `/rest/app-chat/conversations/${conversationId}`) return jsonResponse(conversationMetadata());
      if (url.pathname === `/rest/app-chat/conversations/${conversationId}/responses`) return jsonResponse(responsesPayload());
      if (url.pathname === `/rest/app-chat/conversations/${conversationId}/response-node`) {
        return jsonResponse({ responseNodes: [], inflightResponses: [{ responseId: "r-3", sender: "ASSISTANT" }] });
      }
      throw new Error(`unexpected request: ${url.pathname}`);
    });
    const { requestNative } = installBridge(fetch);

    const result = await requestNative();
    expect(result.capture.ok).toBe(true);
    const body = JSON.parse(result.capture.body);
    expect(body.conversationId).toBe(conversationId);
    expect(body.responses).toHaveLength(2);
    expect(body.responses[1].message).toBe("hi there");
    expect(body.inflightResponses).toEqual([{ responseId: "r-3", sender: "ASSISTANT" }]);
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
    expect(JSON.parse(result.capture.body).inflightResponses).toEqual([]);
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
    const { requestAsset } = installBridge(fetch);

    const result = await requestAsset();
    expect(result.outcome.status).toBe("acquired");
    expect(result.outcome.asset.sha256).toBe(expectedSha256);
    expect(Buffer.from(result.outcome.asset.base64, "base64").toString("utf8")).toBe("polylogue grok asset fixture\n");
  });

  it("reports too_large without buffering past the requested byte budget", async () => {
    const fetch = vi.fn(async () => byteResponse(assetBytes));
    const { requestAsset } = installBridge(fetch);

    const result = await requestAsset({ maxBytes: 4 });
    expect(result.outcome.status).toBe("too_large");
  });

  it("classifies a 403 (verified live: assets.grok.com without credentials) as signed_url_expired", async () => {
    const fetch = vi.fn(async () => new globalThis.Response("", { status: 403 }));
    const { requestAsset } = installBridge(fetch);

    const result = await requestAsset();
    expect(result.outcome.status).toBe("signed_url_expired");
    expect(result.outcome.http_status).toBe(403);
  });
});
