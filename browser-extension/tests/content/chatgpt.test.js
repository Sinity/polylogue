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

// Every request this fixture doesn't explicitly answer 404s immediately
// rather than hanging -- asset acquisition then resolves each descriptor as
// "missing"/"unauthorized" in milliseconds instead of burning its real
// 9s-per-asset timeout, which is what makes it safe to assert on the exact
// descriptor identity (kind/fileId/sandboxPath) these tests care about
// without the suite becoming slow.
function installChatgpt({ url = "https://chatgpt.com/c/conversation-1", fetch } = {}) {
  const dom = new JSDOM("<!doctype html><title>ChatGPT fixture</title>", { url, runScripts: "outside-only" });
  openDoms.push(dom);
  const cryptoAdapter = {
    subtle: {
      digest(algorithm, data) {
        return webcrypto.subtle.digest(algorithm, Buffer.from(new dom.window.Uint8Array(data)));
      },
    },
  };
  Object.defineProperty(dom.window, "crypto", { configurable: true, value: cryptoAdapter });
  Object.defineProperty(dom.window, "fetch", { configurable: true, value: fetch || (async () => notFoundResponse()) });
  const runtimeListeners = [];
  const chrome = {
    runtime: {
      id: "synthetic-extension-id",
      getManifest: () => ({ version: "0.1.0" }),
      onMessage: { addListener: (listener) => runtimeListeners.push(listener) },
      async sendMessage(message) {
        if (message.type === "polylogue.capture") {
          return { ok: true, provider: "chatgpt", provider_session_id: "conversation-1", receiver_request_id: "synthetic-request" };
        }
        if (message.type === "polylogue.archiveState") return { captured: true, state: "archived" };
        return { ok: true };
      },
    },
  };
  Object.defineProperty(dom.window, "chrome", { configurable: true, value: chrome });
  Object.defineProperty(dom.window, "postMessage", {
    configurable: true,
    value(data) {
      dom.window.queueMicrotask(() => {
        dom.window.dispatchEvent(new dom.window.MessageEvent("message", { source: dom.window, origin: dom.window.location.origin, data }));
      });
    },
  });
  const context = dom.getInternalVMContext();
  new Script(bridgeSource).runInContext(context);
  new Script(commonSource).runInContext(context);
  new Script(contentSource).runInContext(context);
  function sendRuntimeMessage(message) {
    return new Promise((resolve, reject) => {
      const listener = runtimeListeners.find((candidate) => candidate(message, {}, resolve) === true);
      if (!listener) reject(new Error(`no runtime listener accepted ${message.type}`));
    });
  }
  return { dom, sendRuntimeMessage };
}

afterEach(() => {
  for (const dom of openDoms.splice(0)) dom.window.close();
});

describe("chatgpt.js on-demand native fetch, exact-provider capture", () => {
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

    expect(result).toMatchObject({ ok: true, envelope: { session: { provider_session_id: "conv-123", turns: [{ text: "hello" }] } } });
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
  });
});

describe("chatgpt.js asset descriptor identification (through a real capture)", () => {
  it("collects sandbox links, upload ids, and asset pointers with parser-matching ids", async () => {
    const fetch = vi.fn(async (input) => {
      const url = new URL(String(input), "https://chatgpt.com");
      if (url.pathname === "/backend-api/conversation/conversation-1") {
        return jsonResponse({
          id: "conversation-1",
          conversation_id: "conversation-1",
          mapping: {
            n1: {
              id: "n1",
              parent: null,
              message: {
                id: "msg-a",
                author: { role: "assistant" },
                content: {
                  content_type: "text",
                  parts: ["Kit ready: [zip](sandbox:/mnt/data/kit.zip) and [sum](sandbox:/mnt/data/kit.zip.sha256). Again sandbox:/mnt/data/kit.zip."],
                },
                metadata: {},
              },
            },
            n2: {
              id: "n2",
              parent: "n1",
              message: {
                id: "msg-b",
                author: { role: "user" },
                content: { content_type: "text", parts: ["please use sandbox:/mnt/data/ignored.zip"] },
                metadata: { attachments: [{ id: "file-UP1", name: "input.csv", mime_type: "text/csv" }] },
              },
            },
            n3: {
              id: "n3",
              parent: "n2",
              message: {
                id: "msg-c",
                author: { role: "assistant" },
                content: { content_type: "multimodal_text", parts: [{ content_type: "image_asset_pointer", asset_pointer: "file-service://file-IMG9" }] },
                metadata: {},
              },
            },
          },
        });
      }
      // Every asset metadata/download round trip 404s -- this test cares
      // about which descriptors were IDENTIFIED, not byte acquisition.
      return notFoundResponse();
    });
    const { sendRuntimeMessage } = installChatgpt({ url: "https://chatgpt.com/c/conversation-1", fetch });

    const result = await sendRuntimeMessage({ type: "polylogue.capturePage", reason: "message_layer_save" });

    expect(result.ok).toBe(true);
    const acquisition = result.envelope.session.provider_meta.asset_acquisition;
    const attemptedIds = [...acquisition.failed, ...acquisition.acquired_assets].map((entry) => entry.provider_attachment_id);
    expect(attemptedIds).toEqual([
      "sandbox:msg-a:/mnt/data/kit.zip",
      "sandbox:msg-a:/mnt/data/kit.zip.sha256",
      "file-UP1",
      "file-service://file-IMG9",
    ]);
    expect(acquisition.attempted).toBe(4);
    // n2's sandbox link is on a user-authored turn, not assistant -- assets
    // never manifest as user prose, so it must not be identified at all.
    expect(attemptedIds).not.toContain("sandbox:msg-b:/mnt/data/ignored.zip");
  });

  it("strips trailing punctuation and dedupes sandbox paths within one turn", async () => {
    const fetch = vi.fn(async (input) => {
      const url = new URL(String(input), "https://chatgpt.com");
      if (url.pathname === "/backend-api/conversation/conversation-1") {
        return jsonResponse({
          id: "conversation-1",
          conversation_id: "conversation-1",
          mapping: {
            n1: {
              id: "n1",
              parent: null,
              message: {
                id: "msg-a",
                author: { role: "assistant" },
                content: { content_type: "text", parts: ["see sandbox:/mnt/data/a.md. and sandbox:/mnt/data/a.md,"] },
                metadata: {},
              },
            },
          },
        });
      }
      return notFoundResponse();
    });
    const { sendRuntimeMessage } = installChatgpt({ url: "https://chatgpt.com/c/conversation-1", fetch });

    const result = await sendRuntimeMessage({ type: "polylogue.capturePage", reason: "message_layer_save" });

    const acquisition = result.envelope.session.provider_meta.asset_acquisition;
    const attemptedIds = [...acquisition.failed, ...acquisition.acquired_assets].map((entry) => entry.provider_attachment_id);
    expect(attemptedIds).toEqual(["sandbox:msg-a:/mnt/data/a.md"]);
  });
});
