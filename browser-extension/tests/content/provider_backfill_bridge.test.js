import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { Script } from "node:vm";
import { TextEncoder } from "node:util";

import { afterEach, describe, expect, it, vi } from "vitest";
import { JSDOM } from "jsdom";

const source = readFileSync(resolve(import.meta.dirname, "../../src/content/provider_backfill_bridge.js"), "utf8");
const openDoms = [];

function jsonResponse(body, status = 200, extraHeaders = {}) {
  const serialized = JSON.stringify(body);
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: { get: (name) => {
      const normalized = name.toLowerCase();
      if (normalized === "content-type") return "application/json";
      return extraHeaders[normalized] || null;
    } },
    body: null,
    text: vi.fn(async () => serialized),
  };
}

function install(url, fetchImpl, storage = {}) {
  const dom = new JSDOM("<!doctype html><title>provider fixture</title>", { url, runScripts: "outside-only" });
  openDoms.push(dom);
  for (const [key, value] of Object.entries(storage)) dom.window.localStorage.setItem(key, value);
  Object.defineProperty(dom.window, "fetch", { configurable: true, value: fetchImpl });
  Object.defineProperty(dom.window, "TextEncoder", { configurable: true, value: TextEncoder });
  const posted = [];
  Object.defineProperty(dom.window, "postMessage", {
    configurable: true,
    value(message) { posted.push(message); },
  });
  new Script(source).runInContext(dom.getInternalVMContext());
  async function request(provider, operation, params = {}) {
    const requestId = `request-${posted.length + 1}`;
    dom.window.dispatchEvent(new dom.window.MessageEvent("message", {
      source: dom.window,
      origin: dom.window.location.origin,
      data: { type: "polylogue.backfill.pageRequest", requestId, provider, operation, params },
    }));
    await vi.waitFor(() => expect(posted.some((entry) => entry.requestId === requestId)).toBe(true));
    return posted.find((entry) => entry.requestId === requestId);
  }
  return { request, posted };
}

afterEach(() => {
  for (const dom of openDoms.splice(0)) dom.window.close();
});

describe("first-party backfill page bridge", () => {
  it("adds ephemeral ChatGPT bearer and selected-account context without disclosing either", async () => {
    const calls = [];
    const token = "synthetic-bearer-secret";
    const accountId = "synthetic-account-secret";
    const fetchImpl = vi.fn(async (input, options = {}) => {
      const url = new URL(input, "https://chatgpt.com");
      calls.push({ url, options });
      if (url.pathname === "/api/auth/session") return jsonResponse({ accessToken: token, account: { id: accountId } });
      const headers = options.headers || {};
      if (headers.Authorization !== `Bearer ${token}` || headers["ChatGPT-Account-Id"] !== accountId) {
        return jsonResponse({ items: [], total: 0 });
      }
      return jsonResponse({ items: [{ id: "conversation-1" }], total: 2 });
    });
    const bridge = install("https://chatgpt.com/", fetchImpl);

    const result = await bridge.request("chatgpt", "inventory", { offset: 0, limit: 1 });

    expect(result.response).toMatchObject({ ok: true, status: 200 });
    expect(JSON.parse(result.response.body)).toMatchObject({ total: 2, items: [{ id: "conversation-1" }] });
    const inventoryCall = calls.find((call) => call.url.pathname === "/backend-api/conversations");
    expect(inventoryCall.url.searchParams.get("is_archived")).toBe("false");
    expect(inventoryCall.url.searchParams.get("is_starred")).toBe("false");
    expect(JSON.stringify(bridge.posted)).not.toContain(token);
    expect(JSON.stringify(bridge.posted)).not.toContain(accountId);
  });

  it("rejects arbitrary operations before network I/O", async () => {
    const fetchImpl = vi.fn();
    const bridge = install("https://chatgpt.com/", fetchImpl);

    const result = await bridge.request("chatgpt", "arbitrary", { url: "https://example.test/" });

    expect(result.error).toBe("backfill_bridge_operation_not_allowed");
    expect(fetchImpl).not.toHaveBeenCalled();
  });

  it("forwards only Retry-After provider metadata needed by the scheduler", async () => {
    const fetchImpl = vi.fn(async (input) => {
      const url = new URL(input, "https://chatgpt.com");
      if (url.pathname === "/api/auth/session") return jsonResponse({ accessToken: "token", account: { id: "account" } });
      return jsonResponse({ detail: "slow down" }, 429, { "retry-after": "60", "set-cookie": "secret" });
    });
    const bridge = install("https://chatgpt.com/", fetchImpl);

    const result = await bridge.request("chatgpt", "inventory", { offset: 0, limit: 28 });

    expect(result.response).toMatchObject({ status: 429, retryAfter: "60" });
    expect(result.response).not.toHaveProperty("headers");
    expect(JSON.stringify(result)).not.toContain("set-cookie");
  });

  it("orders Claude organizations by the UI-selected organization without persisting the selector", async () => {
    const selected = "22222222-2222-4222-8222-222222222222";
    const other = "11111111-1111-4111-8111-111111111111";
    const fetchImpl = vi.fn(async () => jsonResponse([{ uuid: other }, { uuid: selected }]));
    const bridge = install("https://claude.ai/new", fetchImpl, {
      [`claude-mcp-has-connectors:${other}`]: "true",
      [`claude-mcp-has-connectors:${selected}`]: "true",
      "omelette-org-settings-cache": JSON.stringify({ v: 1, orgUuid: selected, settings: {} }),
    });

    const result = await bridge.request("claude-ai", "organizations");

    expect(JSON.parse(result.response.body).map((entry) => entry.uuid)).toEqual([selected, other]);
    expect(JSON.stringify(bridge.posted)).not.toContain("claude-mcp-has-connectors");
  });
});
