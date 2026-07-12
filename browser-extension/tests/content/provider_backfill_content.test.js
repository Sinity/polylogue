import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { Script } from "node:vm";

import { afterEach, describe, expect, it, vi } from "vitest";
import { JSDOM } from "jsdom";

const source = readFileSync(resolve(import.meta.dirname, "../../src/content/provider_backfill_content.js"), "utf8");
const openDoms = [];

function install(responder) {
  const dom = new JSDOM("<!doctype html>", { url: "https://chatgpt.com/", runScripts: "outside-only" });
  openDoms.push(dom);
  let listener;
  const chrome = {
    runtime: {
      id: "extension-id",
      onMessage: { addListener(value) { listener = value; } },
    },
  };
  Object.defineProperty(dom.window, "chrome", { configurable: true, value: chrome });
  Object.defineProperty(dom.window, "crypto", { configurable: true, value: { randomUUID: () => "correlated-request-id" } });
  Object.defineProperty(dom.window, "postMessage", {
    configurable: true,
    value(message) {
      responder?.(message, dom.window);
    },
  });
  new Script(source).runInContext(dom.getInternalVMContext());
  return { dom, listener };
}

afterEach(() => {
  vi.useRealTimers();
  for (const dom of openDoms.splice(0)) dom.window.close();
});

describe("isolated backfill content bridge", () => {
  it("correlates one MAIN-world response without exposing extension authority to the page", async () => {
    const harness = install((message, window) => {
      window.queueMicrotask(() => window.dispatchEvent(new window.MessageEvent("message", {
        source: window,
        origin: window.location.origin,
        data: {
          type: "polylogue.backfill.pageResponse",
          requestId: message.requestId,
          response: { ok: true, status: 200, contentType: "application/json", body: "{}" },
        },
      })));
    });
    const response = await new Promise((resolve) => {
      const keepAlive = harness.listener({ type: "polylogue.backfill.pageRequest", provider: "chatgpt", operation: "inventory" }, { id: "extension-id" }, resolve);
      expect(keepAlive).toBe(true);
    });

    expect(response).toEqual({ ok: true, response: { ok: true, status: 200, contentType: "application/json", body: "{}" } });
  });

  it("rejects a runtime sender outside this extension", async () => {
    const harness = install();
    const response = await new Promise((resolve) => {
      const keepAlive = harness.listener({ type: "polylogue.backfill.pageRequest" }, { id: "other-extension" }, resolve);
      expect(keepAlive).toBe(false);
    });

    expect(response).toEqual({ ok: false, error: "backfill_bridge_sender_mismatch" });
  });
});
