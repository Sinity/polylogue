import { afterEach, describe, expect, it, vi } from "vitest";

import { executeProviderPageRequest } from "../src/backfill/page_transport.js";

const { Headers, Response } = globalThis;
const originalWindow = globalThis.window;

function installWindow(url, fetchImpl, storage = {}) {
  const values = new Map(Object.entries(storage));
  globalThis.window = {
    location: new URL(url),
    fetch: fetchImpl,
    localStorage: { getItem: (key) => values.get(key) || null },
    setTimeout: globalThis.setTimeout.bind(globalThis),
    clearTimeout: globalThis.clearTimeout.bind(globalThis),
  };
}

afterEach(() => {
  globalThis.window = originalWindow;
  vi.restoreAllMocks();
});

describe("first-party provider page transport", () => {
  it("keeps ChatGPT bearer and selected account inside MAIN-world execution", async () => {
    const calls = [];
    const token = "synthetic-bearer-secret";
    const accountId = "synthetic-account-secret";
    const fetchImpl = vi.fn(async (input, options = {}) => {
      const url = new URL(input);
      calls.push({ url, options });
      if (url.pathname === "/api/auth/session") return new Response(JSON.stringify({ accessToken: token, account: { id: accountId } }), { headers: { "Content-Type": "application/json" } });
      const headers = new Headers(options.headers);
      if (headers.get("Authorization") !== `Bearer ${token}` || headers.get("ChatGPT-Account-Id") !== accountId) {
        return new Response(JSON.stringify({ items: [], total: 0 }), { headers: { "Content-Type": "application/json" } });
      }
      return new Response(JSON.stringify({ items: [{ id: "conversation-1" }], total: 2 }), { headers: { "Content-Type": "application/json" } });
    });
    installWindow("https://chatgpt.com/", fetchImpl);

    const result = await executeProviderPageRequest({
      provider: "chatgpt",
      operation: "inventory",
      params: { offset: 0, limit: 1, archived: false, starred: false },
      maxResponseBytes: 4096,
    });

    expect(result).toMatchObject({ ok: true, response: { ok: true, status: 200 } });
    expect(JSON.parse(result.response.body)).toMatchObject({ total: 2, items: [{ id: "conversation-1" }] });
    expect(JSON.stringify(result)).not.toContain(token);
    expect(JSON.stringify(result)).not.toContain(accountId);
    expect(calls.find((call) => call.url.pathname === "/backend-api/conversations").url.searchParams.get("is_archived")).toBe("false");
  });

  it("cancels a chunked response as soon as the byte cap is crossed", async () => {
    const cancel = vi.fn(async () => undefined);
    let reads = 0;
    const response = {
      ok: true,
      status: 200,
      headers: { get: () => null },
      body: { getReader: () => ({
        read: vi.fn(async () => {
          reads += 1;
          if (reads === 1) return { done: false, value: new Uint8Array(6) };
          if (reads === 2) return { done: false, value: new Uint8Array(6) };
          return { done: true };
        }),
        cancel,
      }) },
    };
    const selected = "22222222-2222-4222-8222-222222222222";
    installWindow("https://claude.ai/new", vi.fn(async () => response), {
      "omelette-org-settings-cache": JSON.stringify({ orgUuid: selected, settings: {} }),
    });

    const result = await executeProviderPageRequest({ provider: "claude-ai", operation: "organizations", params: {}, maxResponseBytes: 8 });

    expect(result.error).toMatch(/^backfill_bridge_response_too_large:observed_bytes=12;limit_bytes=8$/);
    expect(cancel).toHaveBeenCalledTimes(1);
    expect(reads).toBe(2);
  });

  it("projects a declared-over-32-MiB ChatGPT conversation before crossing the bridge", async () => {
    const token = "synthetic-bearer-secret";
    const accountId = "synthetic-account-secret";
    const fetchImpl = vi.fn(async (input) => {
      const url = new URL(input);
      if (url.pathname === "/api/auth/session") return new Response(JSON.stringify({ accessToken: token, account: { id: accountId } }));
      return new Response(JSON.stringify({
        id: "conversation-1",
        title: "Compact fixture",
        mapping: {
          node: { id: "node", parent: null, message: { id: "message", author: { role: "assistant" }, content: { parts: ["kept"] }, metadata: { model_slug: "fixture" }, create_time: 1 } },
        },
        ignored_large_provider_metadata: "x".repeat(4096),
      }), { headers: { "Content-Length": String(32 * 1024 * 1024 + 1) } });
    });
    installWindow("https://chatgpt.com/", fetchImpl);

    const result = await executeProviderPageRequest({ provider: "chatgpt", operation: "conversation", params: { nativeId: "conversation-1" }, maxResponseBytes: 32 * 1024 * 1024 });

    expect(result).toMatchObject({ ok: true, response: { ok: true } });
    const body = JSON.parse(result.response.body);
    expect(body).toMatchObject({ polylogue_bridge_projection: "chatgpt-native-compact-v1", mapping: { node: { message: { content: { parts: ["kept"] } } } } });
    expect(JSON.stringify(body)).not.toContain("ignored_large_provider_metadata");
  });

  it("fails closed when a compact ChatGPT projection still exceeds its bridge limit", async () => {
    const token = "synthetic-bearer-secret";
    const accountId = "synthetic-account-secret";
    const fetchImpl = vi.fn(async (input) => {
      const url = new URL(input);
      if (url.pathname === "/api/auth/session") return new Response(JSON.stringify({ accessToken: token, account: { id: accountId } }));
      return new Response(JSON.stringify({
        id: "conversation-1",
        mapping: {
          node: { id: "node", parent: null, message: { id: "message", author: { role: "assistant" }, content: { parts: ["x".repeat(8 * 1024 * 1024)] } } },
        },
      }));
    });
    installWindow("https://chatgpt.com/", fetchImpl);

    const result = await executeProviderPageRequest({ provider: "chatgpt", operation: "conversation", params: { nativeId: "conversation-1" }, maxResponseBytes: 32 * 1024 * 1024 });

    expect(result.error).toMatch(/^backfill_bridge_projection_too_large:observed_bytes=.+;limit_bytes=8388608$/);
  });

  it("uses the exact Claude UI selector despite ambiguous per-organization keys", async () => {
    const selected = "22222222-2222-4222-8222-222222222222";
    const other = "11111111-1111-4111-8111-111111111111";
    installWindow("https://claude.ai/new", vi.fn(async () => new Response(JSON.stringify([{ uuid: other }, { uuid: selected }]), { headers: { "Content-Type": "application/json", "Retry-After": "60" } })), {
      [`claude-mcp-has-connectors:${other}`]: "true",
      [`claude-mcp-has-connectors:${selected}`]: "true",
      "omelette-org-settings-cache": JSON.stringify({ orgUuid: selected, settings: {} }),
    });

    const result = await executeProviderPageRequest({ provider: "claude-ai", operation: "organizations", params: {}, maxResponseBytes: 4096 });

    expect(JSON.parse(result.response.body).map((entry) => entry.uuid)).toEqual([selected, other]);
    expect(result.response.retryAfter).toBe("60");
    expect(result.response).not.toHaveProperty("headers");
  });
});
